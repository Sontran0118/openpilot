#!/usr/bin/env python3
"""SUPERCOMBO -> modelV2 PUBLISHER shim.

Runs the real supercombo model on live IMX477 frames and publishes its output
onto the cereal bus as `modelV2`, so controlsd consumes the REAL model instead
of synthetic data. This is the missing link between the model port and the
control stack.

Mapping (SupercomboRunner.step() -> modelV2):
  path_xyz (33,3)      -> position.x/y/z + t
  lane_lines (4,33,2)  -> laneLines[i].x/y/z
  road_edges (2,33,2)  -> roadEdges[i].x/y/z
  path curvature       -> action.desiredCurvature (what controlsd steers on)

Read-only + publish. No transmit to car.
Usage: python3 supercombo_publisher.py [--synthetic] [-n N]
"""
import sys, os, time, math, subprocess

os.environ.setdefault("PARAMS_ROOT", "/tmp/op_params")
sys.path.insert(0, "/home/tran/openpilot_jetson")

import numpy as np
import openpilot.cereal.messaging as messaging
sys.path.insert(0,"/home/tran/openpilot_jetson")
from curvature_lib import path_to_curvature

# openpilot model time indices (T_IDXS) — 33 points, matches supercombo path
T_IDXS = [0.0, 0.00976, 0.0398, 0.0898, 0.159, 0.248, 0.357, 0.485, 0.633,
          0.801, 0.989, 1.197, 1.425, 1.673, 1.941, 2.229, 2.537, 2.865,
          3.213, 3.581, 3.969, 4.377, 4.805, 5.253, 5.721, 6.209, 6.717,
          7.245, 7.793, 8.361, 8.949, 9.557, 10.185]


def path_curvature(path_xyz, v_ego=20.0):
    return path_to_curvature(path_xyz, v_ego)


def fill_modelV2(md, out):
    path = out.get("path_xyz")
    lanes = out.get("lane_lines")
    edges = out.get("road_edges")
    N = 33

    md.frameId = fill_modelV2.frame
    fill_modelV2.frame += 1

    if path is not None:
        px = [float(path[i][0]) for i in range(N)]
        py = [float(path[i][1]) for i in range(N)]
        pz = [float(path[i][2]) for i in range(N)]
        md.position.x = px
        md.position.y = py
        md.position.z = pz
        md.position.t = T_IDXS

    # lane lines (4)
    if lanes is not None:
        ll = md.init('laneLines', 4)
        for k in range(4):
            ll[k].x = [float(lanes[k][i][0]) for i in range(N)]
            ll[k].y = [float(lanes[k][i][1]) for i in range(N)]
            ll[k].z = [0.0] * N
            ll[k].t = T_IDXS
        lp = out.get("lane_prob")
        if lp is not None:
            md.laneLineProbs = [float(x) for x in lp[:4]] if len(lp) >= 4 else [0.5]*4

    # road edges (2)
    if edges is not None:
        re = md.init('roadEdges', 2)
        for k in range(2):
            re[k].x = [float(edges[k][i][0]) for i in range(N)]
            re[k].y = [float(edges[k][i][1]) for i in range(N)]
            re[k].z = [0.0] * N
            re[k].t = T_IDXS

    # the field controlsd steers on
    try:
        md.action.desiredCurvature = float(path_curvature(path))
    except Exception:
        pass


fill_modelV2.frame = 0


def grab_frame():
    subprocess.run(
        ["gst-launch-1.0", "-q", "nvarguscamerasrc", "num-buffers=1",
         "!", "video/x-raw(memory:NVMM),width=1280,height=720",
         "!", "nvvidconv", "!", "jpegenc", "!", "filesink", "location=/tmp/vf.jpg"],
        capture_output=True, timeout=15)
    import cv2
    return cv2.imread("/tmp/vf.jpg")


def main():
    synthetic = "--synthetic" in sys.argv
    n = 10
    if "-n" in sys.argv:
        n = int(sys.argv[sys.argv.index("-n") + 1])

    pm = messaging.PubMaster(['modelV2'])
    sm = messaging.SubMaster(['modelV2'])

    runner = None
    if not synthetic:
        from op_stream import SupercomboRunner
        runner = SupercomboRunner()
        print("supercombo loaded — publishing REAL model output as modelV2")
    else:
        print("synthetic mode — publishing a curved fake path")

    time.sleep(0.5)
    for i in range(n):
        if synthetic:
            V = 20.0
            fake_path = np.array([[j * V * 0.1, 0.002 * j * j, 0.0] for j in range(33)])
            out = {"path_xyz": fake_path, "lane_lines": None, "road_edges": None}
        else:
            frame = grab_frame()
            if frame is None:
                print("  (no frame)"); continue
            out = runner.step(frame)

        msg = messaging.new_message('modelV2')
        fill_modelV2(msg.modelV2, out)
        pm.send('modelV2', msg)

        # confirm it's on the bus + show the curvature controlsd will use
        time.sleep(0.05)
        sm.update(50)
        curv = msg.modelV2.action.desiredCurvature
        print("frame %2d: published modelV2, desiredCurvature=%.5f  (received=%s)" %
              (i, curv, sm.updated['modelV2']))

    print("\n=== supercombo -> modelV2 on the bus. controlsd can now consume it. ===")


if __name__ == "__main__":
    main()
