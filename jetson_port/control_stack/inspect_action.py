import sys, cv2, numpy as np
sys.path.insert(0, "/home/tran/openpilot_jetson")
from op_stream import SupercomboRunner

r = SupercomboRunner()
f = cv2.imread("/tmp/vf.jpg")
out = r.step(f)

print("=== step() output keys ===")
for k, v in out.items():
    shape = getattr(v, "shape", None)
    print("  %-16s %s" % (k, shape if shape is not None else type(v).__name__))

# Does the runner expose the RAW output vector? Check for action / desired_curvature.
print("\n=== looking for curvature/action in output ===")
for key in out:
    if "curv" in key.lower() or "action" in key.lower() or "desire" in key.lower():
        print("  FOUND:", key, "=", out[key])

# The supercombo output_slices metadata names every field. Print any the runner kept.
for attr in ("output_slices", "raw", "raw_output", "outputs", "action", "desired_curvature"):
    if hasattr(r, attr):
        v = getattr(r, attr)
        print("  runner.%s: %s" % (attr, type(v).__name__))

# Path-derived vs any direct field: show the path near the car for comparison
if "path_xyz" in out:
    p = out["path_xyz"]
    print("\npath_xyz first 3:", p[:3].tolist())
    # openpilot-style curvature from the path (orientation-based would need orientation output)
    i = 10
    x, y = float(p[i][0]), float(p[i][1])
    print("path lookahead[%d] x=%.2f y=%.3f -> approx curv 2y/x^2 = %.5f" %
          (i, x, y, (2*y/(x*x)) if abs(x) > 1 else 0.0))

# Is there an orientation output we should use for a proper curvature?
if "orientation" in out:
    print("\norientation present:", out["orientation"].shape)
