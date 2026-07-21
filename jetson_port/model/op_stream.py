#!/usr/bin/env python3
"""Complete streaming openpilot supercombo runner on the Jetson — WITH recurrent feedback.
Mirrors openpilot modeld: rolling feature queue (frame_skip=4 x 24 = 96 deep), the model's
hidden_state fed back each frame, subsampled to the 24-frame features_buffer. Processes a stream
of frames (from disk or camera) and decodes the driving plan each step.

  class SupercomboRunner: .step(bgr_frame) -> dict(path_xyz, lane_lines, lead_prob, pose, ...)
"""
import numpy as np, tensorrt as trt, torch, sys, base64, pickle, collections
sys.path.insert(0, "/home/tran/openpilot_jetson"); sys.path.insert(0, "/home/tran/openpilot_jetson/modeld")
from op_frame import frame_to_model_input
from constants import ModelConstants as MC, Plan

ENGINE="/home/tran/openpilot_jetson/models/supercombo_fp32.trt"
ONNX="/home/tran/openpilot_jetson/models/driving_supercombo.onnx"
FRAME_SKIP = MC.MODEL_RUN_FREQ // MC.MODEL_CONTEXT_FREQ    # 20//5 = 4
FB_LEN = 24                                                # features_buffer temporal length

def softmax(x,axis=-1): x=x-x.max(axis,keepdims=True); e=np.exp(x); return e/e.sum(axis,keepdims=True)
def sigmoid(x): return 1/(1+np.exp(-x))
def safe_exp(x): return np.exp(np.clip(x,-np.inf,11))

def load_slices():
    import onnx
    m=onnx.load(ONNX, load_external_data=False)
    for p in m.metadata_props:
        if p.key=="output_slices": return pickle.loads(base64.b64decode(p.value))

class SupercomboRunner:
    def __init__(self):
        with open(ENGINE,"rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
            self.eng=rt.deserialize_cuda_engine(f.read())
        self.ctx=self.eng.create_execution_context(); self.dev="cuda"; self.st=torch.cuda.Stream()
        self.slices=load_slices()
        # recurrent feature queue: FRAME_SKIP*FB_LEN deep (96), each (512,), newest last
        self.feat_q=collections.deque([np.zeros(MC.FEATURE_LEN,np.float32)]*(FRAME_SKIP*FB_LEN),
                                      maxlen=FRAME_SKIP*FB_LEN)
        self.prev_yuv=None
        # bind IO tensors once
        self.io={}; self.out_name=None
        for i in range(self.eng.num_io_tensors):
            n=self.eng.get_tensor_name(i); shp=tuple(self.eng.get_tensor_shape(n))
            dt=torch.float16 if self.eng.get_tensor_dtype(n)==trt.float16 else (
               torch.uint8 if self.eng.get_tensor_dtype(n)==trt.uint8 else torch.float32)
            t=torch.zeros(shp,dtype=dt,device=self.dev); self.io[n]=t
            self.ctx.set_tensor_address(n,t.data_ptr())
            if self.eng.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT: self.out_name=n

    def _features_buffer(self):
        # subsample the deep queue every FRAME_SKIP -> (24,512) -> (1,24,512)
        arr=np.stack(list(self.feat_q),0)[::FRAME_SKIP]         # (24,512)
        return arr[None].astype(np.float16)

    @torch.no_grad()
    def step(self, bgr, desire=None, traffic=(1,0)):
        img12,self.prev_yuv=frame_to_model_input(bgr, prev_yuv=self.prev_yuv)
        feed={"img":img12.astype(np.uint8),"big_img":img12.astype(np.uint8),
              "features_buffer":self._features_buffer(),
              "desire_pulse":(desire if desire is not None else np.zeros((1,25,8),np.float16)),
              "traffic_convention":np.array([traffic],np.float16),"action_t":np.zeros((1,2),np.float16)}
        for n,t in self.io.items():
            if n in feed: t.copy_(torch.from_numpy(np.ascontiguousarray(feed[n])).to(self.dev))
        with torch.cuda.stream(self.st): self.ctx.execute_async_v3(self.st.cuda_stream)
        self.st.synchronize()
        out=self.io[self.out_name].float().cpu().numpy().reshape(-1)
        # FEED BACK: append this frame's hidden_state to the recurrent queue
        hs=out[self.slices['hidden_state']].astype(np.float32)
        self.feat_q.append(hs)
        return self._decode(out)

    def _decode(self,out):
        # MDN outputs pack [mu (n), std (n)]; take the mu half. plan 990 -> mu 495 = 33x15.
        o=out.astype(np.float32); s=self.slices
        def mu(name, per): return o[s[name]][:per]
        plan=mu('plan', MC.IDX_N*MC.PLAN_WIDTH).reshape(MC.IDX_N,MC.PLAN_WIDTH)
        ll=mu('lane_lines', 4*MC.IDX_N*2).reshape(4,MC.IDX_N,2)
        re_=mu('road_edges', 2*MC.IDX_N*2).reshape(2,MC.IDX_N,2)
        return dict(
            path_xyz=plan[:,Plan.POSITION],
            path_vel=plan[:,Plan.VELOCITY],
            lane_lines=ll, road_edges=re_,
            lane_prob=sigmoid(o[s['lane_lines_prob']]),
            lead_prob=sigmoid(o[s['lead_prob']]),
            pose=o[s['pose']][:6],
            hidden_state=o[s['hidden_state']],
        )

if __name__=="__main__":
    import cv2,glob,time
    r=SupercomboRunner()
    # stream: replay /tmp/hd.jpg N times to show the recurrent buffer filling + timing
    frame=cv2.imread("/tmp/hd.jpg")
    if frame is None: frame=(np.random.rand(720,1280,3)*255).astype(np.uint8)
    print(f"FRAME_SKIP={FRAME_SKIP}, feat_q depth={FRAME_SKIP*FB_LEN}, features_buffer=(1,{FB_LEN},512)")
    t0=time.time(); N=60
    for i in range(N):
        d=r.step(frame)
    hz=N/(time.time()-t0)
    print(f"streaming with recurrent feedback: {hz:.1f} Hz over {N} frames")
    print(f"path reach: {d['path_xyz'][-1,0]:.1f}m  final lateral: {d['path_xyz'][-1,1]:+.2f}m")
    print(f"lane_prob: {d['lane_prob'].round(2)}  lead_prob: {d['lead_prob'].round(2)}")
    print(f"hidden_state norm (should be nonzero + evolving): {np.linalg.norm(d['hidden_state']):.2f}")
    # verify the recurrent queue actually filled with real features (not zeros)
    qnorm=np.linalg.norm(np.stack(list(r.feat_q),0),axis=1)
    print(f"feat_q norms: first={qnorm[0]:.1f} last={qnorm[-1]:.1f} (last should be >0 = recurrent working)")
    print("RECURRENT TEMPORAL MODEL COMPLETE: hidden_state feeds back into features_buffer.")
