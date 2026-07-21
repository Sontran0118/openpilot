#!/usr/bin/env python3
"""Port of openpilot's camera->model-input transform, standalone for the Jetson + IMX477.
Produces the model's 'img'/'big_img' [1,12,128,256] uint8 input from a camera frame.

Pipeline (matches openpilot exactly):
  1) get_warp_matrix(calib_euler, cam_intrinsics) : model-frame -> camera-pixel perspective matrix
  2) cv2.warpPerspective the camera YUV into the model's 512x256 calibrated frame
  3) pack the warped YUV420 into 12 channels: 2 frames x (4 Y-quadrants + U + V)
"""
import numpy as np, cv2

# --- openpilot model geometry (from common/transformations/model.py) ---
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_CY = 47.6
medmodel_fl = 910.0
medmodel_intrinsics = np.array([
    [medmodel_fl, 0.0, 0.5*MEDMODEL_INPUT_SIZE[0]],
    [0.0, medmodel_fl, MEDMODEL_CY],
    [0.0, 0.0, 1.0]])

# device<->view frame (camera.py)
device_frame_from_view_frame = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
view_frame_from_device_frame = device_frame_from_view_frame.T

def rot_from_euler(rpy):
    r,p,y = rpy
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def get_view_frame_from_calib_frame(roll,pitch,yaw,height):
    device_from_calib = rot_from_euler([roll,pitch,yaw])
    view_from_calib = view_frame_from_device_frame.dot(device_from_calib)
    return np.hstack((view_from_calib, [[0],[height],[0]]))

medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics, get_view_frame_from_calib_frame(0,0,0,0))
calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])

def get_warp_matrix(device_from_calib_euler, intrinsics):
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    return camera_from_calib @ calib_from_medmodel   # model-frame -> camera-pixel

def imx477_intrinsics(cap_w, cap_h):
    """Approx pinhole intrinsics for the IMX477 at capture size. The IMX477 sensor is 7.9mm,
    ArduCam UC-517 with a wide lens; use fx ~= cap_w * 0.7 as a starting focal (tune from calibration).
    Principal point = image center."""
    fl = cap_w * 0.70
    return np.array([[fl,0.0,cap_w/2.0],[0.0,fl,cap_h/2.0],[0.0,0.0,1.0]])

def frame_to_model_input(bgr, calib_euler=(0.0,0.0,0.0), prev_yuv=None):
    """bgr: HxWx3 camera frame. Returns (img12 [1,12,128,256] uint8, this_yuv) — pass this_yuv back as
    prev_yuv next call for the 2-frame temporal stack."""
    H, W = bgr.shape[:2]
    K = imx477_intrinsics(W, H)
    M = get_warp_matrix(calib_euler, K)                       # model->camera
    # warp camera frame INTO the model's 512x256 view (need camera->model = inv(M)... warpPerspective
    # maps dst<-src with the matrix mapping src->dst, so we pass M mapping model(dst)->camera(src) with WARP_INVERSE_MAP)
    warped = cv2.warpPerspective(bgr, M, MEDMODEL_INPUT_SIZE,
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(0,0,0))
    yuv = cv2.cvtColor(warped, cv2.COLOR_BGR2YUV_I420)        # (256*1.5, 512)
    y = yuv[:256,:]
    u = yuv[256:256+64,:].reshape(128,256)
    v = yuv[256+64:256+128,:].reshape(128,256)
    # 6 channels for this frame: 4 Y quadrants + U + V
    six = np.stack([y[0::2,0::2], y[0::2,1::2], y[1::2,0::2], y[1::2,1::2], u, v],0).astype(np.uint8)
    prev = prev_yuv if prev_yuv is not None else six          # first frame: duplicate
    img12 = np.concatenate([prev, six],0)[None]               # (1,12,128,256)
    return img12, six

if __name__ == "__main__":
    import os
    p = "/tmp/hd.jpg"
    bgr = cv2.imread(p) if os.path.isfile(p) else (np.random.rand(720,1280,3)*255).astype(np.uint8)
    img12, yuv = frame_to_model_input(bgr)
    print("input frame:", bgr.shape, "-> model img12:", img12.shape, img12.dtype,
          "range", img12.min(), img12.max())
    # also dump the warped model-view so we can SEE the calibrated frame
    K = imx477_intrinsics(bgr.shape[1], bgr.shape[0]); M = get_warp_matrix((0,0,0), K)
    warped = cv2.warpPerspective(bgr, M, MEDMODEL_INPUT_SIZE, flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    cv2.imwrite("/tmp/model_view.jpg", warped)
    print("wrote /tmp/model_view.jpg (what the model sees)")
