#!/usr/bin/env python3
"""Decode the supercombo 2576-float output into the driving plan using openpilot's own Parser.
Extracts: driving PATH (33 xyz points), lane lines, road edges, lead car, pose, and the recurrent
hidden_state to feed back. Slices come from the ONNX's embedded output_slices metadata."""
import numpy as np, sys, base64, pickle
sys.path.insert(0, "/home/tran/openpilot_jetson/modeld")
sys.path.insert(0, "/home/tran/openpilot_jetson")

# helpers openpilot's parser needs (softmax/sigmoid/safe_exp) - self-contained
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e/e.sum(axis=axis, keepdims=True)
def sigmoid(x): return 1/(1+np.exp(-x))
def safe_exp(x): return np.exp(np.clip(x, -np.inf, 11))
# inject into helpers namespace so parse_model_outputs imports resolve
import types
helpers = types.ModuleType("helpers"); helpers.softmax=softmax; helpers.sigmoid=sigmoid; helpers.safe_exp=safe_exp
sys.modules["openpilot.selfdrive.modeld.helpers"]=helpers

from constants import ModelConstants as MC

def load_slices(onnx_path):
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    for p in m.metadata_props:
        if p.key == "output_slices":
            return pickle.loads(base64.b64decode(p.value))
    raise RuntimeError("no output_slices")

# minimal Parser (copied logic from openpilot parse_model_outputs.Parser)
class Parser:
    def parse_mdn(self, name, outs, in_N=0, out_N=1, out_shape=()):
        raw = outs[name]; raw = raw.reshape((raw.shape[0], max(in_N,1), -1))
        n = (raw.shape[2]-out_N)//2
        mu = raw[:,:,:n]; std = safe_exp(raw[:,:,n:2*n])
        if out_N>1: final=tuple([raw.shape[0],out_N]+list(out_shape))
        else: final=tuple([raw.shape[0]]+list(out_shape))
        outs[name]=mu.reshape(final); outs[name+"_stds"]=std.reshape(final)
    def parse_bce(self,name,outs): outs[name]=sigmoid(outs[name])
    def parse_cce(self,name,outs,out_shape=None):
        raw=outs[name]
        if out_shape is not None: raw=raw.reshape((raw.shape[0],)+out_shape)
        outs[name]=softmax(raw,axis=-1)

def decode(out2576, slices):
    o = out2576.reshape(1,-1).astype(np.float32)
    outs = {name:o[:, s] for name,s in slices.items() if isinstance(s,slice)}
    p = Parser()
    p.parse_mdn('pose',outs,0,0,(MC.POSE_WIDTH,))
    p.parse_mdn('lane_lines',outs,0,0,(MC.NUM_LANE_LINES,MC.IDX_N,MC.LANE_LINES_WIDTH))
    p.parse_mdn('road_edges',outs,0,0,(MC.NUM_ROAD_EDGES,MC.IDX_N,MC.LANE_LINES_WIDTH))
    p.parse_mdn('plan',outs,0,0,(MC.IDX_N,MC.PLAN_WIDTH))
    p.parse_bce('lane_lines_prob',outs); p.parse_bce('lead_prob',outs); p.parse_bce('meta',outs)
    return outs

if __name__=="__main__":
    ONNX="/home/tran/openpilot_jetson/models/driving_supercombo.onnx"
    slices=load_slices(ONNX)
    # run one real inference through the pipeline to get a live 2576 vector
    import tensorrt as trt, torch, cv2
    from op_frame import frame_to_model_input
    with open("/home/tran/openpilot_jetson/models/supercombo_fp32.trt","rb") as f, trt.Runtime(trt.Logger(trt.Logger.ERROR)) as rt:
        eng=rt.deserialize_cuda_engine(f.read())
    ctx=eng.create_execution_context(); dev="cuda"; st=torch.cuda.Stream()
    frame=cv2.imread("/tmp/hd.jpg"); frame=frame if frame is not None else (np.random.rand(720,1280,3)*255).astype(np.uint8)
    img12,_=frame_to_model_input(frame)
    feed={"img":img12.astype(np.uint8),"big_img":img12.astype(np.uint8),
          "features_buffer":np.zeros((1,24,512),np.float16),"desire_pulse":np.zeros((1,25,8),np.float16),
          "traffic_convention":np.array([[1,0]],np.float16),"action_t":np.zeros((1,2),np.float16)}
    io={}
    for i in range(eng.num_io_tensors):
        n=eng.get_tensor_name(i)
        if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
            t=torch.from_numpy(np.ascontiguousarray(feed[n])).to(dev); io[n]=t; ctx.set_tensor_address(n,t.data_ptr())
        else:
            out=torch.zeros(tuple(eng.get_tensor_shape(n)),dtype=torch.float16,device=dev); io[n]=out; ctx.set_tensor_address(n,out.data_ptr()); on=n
    with torch.cuda.stream(st): ctx.execute_async_v3(st.cuda_stream)
    st.synchronize()
    out=io[on].float().cpu().numpy().reshape(-1)
    d=decode(out,slices)
    print("=== DECODED DRIVING OUTPUTS ===")
    plan=d['plan'][0]                                  # (33,15)
    path_xyz=plan[:, MC.POSITION]                       # (33,3) forward,left,up in meters
    print(f"PATH (ego trajectory), first 6 of 33 points [fwd, left, up] meters:")
    for i in range(0,12,2): print(f"   t={MC.T_IDXS[i]:.1f}s: {path_xyz[i].round(2)}")
    print(f"path forward reach: {path_xyz[-1,0]:.1f} m,  final lateral: {path_xyz[-1,1]:+.2f} m")
    print(f"lane_line_probs: {d['lane_lines_prob'][0].round(2)}")
    print(f"lead_prob: {d['lead_prob'][0].round(2)}")
    ll=d['lane_lines'][0]                               # (4,33,2)
    print(f"lane_lines shape {ll.shape} (4 lines x 33 pts x [y,z])")
    print(f"ego pose (v/rot): {d['pose'][0].round(3)}")
    print("\nDECODE COMPLETE: 2576 floats -> path + lanes + lead + pose.")
