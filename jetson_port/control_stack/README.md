# control_stack/ — openpilot control-and-safety stack brought up on the Jetson Orin

Scripts that stand up the real openpilot lateral pipeline on the Jetson for the
Mazda CX-5 2023, from the supercombo model through controlsd to a steering-torque
command. Everything here is **read-only / compute / print-only** — none of it
transmits to the car.

## Run environment (required for all scripts)

```bash
export PATH=$HOME/.local/bin:$PATH
export PARAMS_ROOT=/tmp/op_params
export PYTHONPATH=/home/tran/msgq_build:/home/tran/opendbc_src:/home/tran/op_fork:/home/tran/op_fork/openpilot
mkdir -p /tmp/op_params
```

Prereqs (one-time):
- `sudo apt install capnproto libcapnp-dev libzmq3-dev`
- `pip install --break-system-packages Cython cffi crcmod pyzmq setproctitle tinygrad`
- Build msgq: `git clone https://github.com/commaai/msgq.git ~/msgq_build && cd ~/msgq_build && scons -j4`
- Build cereal capnp C++ gen: `bash gen_capnp.sh`
- Build Params: `bash build_params.sh` (uses swaglog_stub.cc to avoid zmq/json11 chain)

## The pipeline (in dependency order)

| script | what it does |
|---|---|
| `test_bus.py` | verify the cereal message bus works (publish modelV2, receive it) |
| `gen_capnp.sh` | capnp -> C++ codegen for cereal (log/car/deprecated/custom) |
| `build_params.sh` + `swaglog_stub.cc` | compile common/params_pyx.so |
| `override_test2.py` | **safety regression test** — runs steering cmds through the REAL safety_mazda C code (libsafety). 16/16. Confirms disengaged->blocked, over-limit->blocked, rate->blocked, driver override widens envelope. |
| `init_controlsd.py` | build CX-5 CarParams, write to params store, instantiate `Controls()` |
| `run_controlsd_step.py` | drive controlsd through ONE control step with synthetic inputs -> prints the steering torque |
| `curvature_lib.py` | `path_to_curvature()` — proper path->curvature (quadratic fit, validated vs known arcs). supercombo outputs PATH not curvature. |
| `supercombo_publisher.py` | run supercombo on live IMX477 -> publish modelV2 onto the bus |
| `virtual_lkas.py` | model path -> 0x243 CAM_LKAS frame (counter+checksum), PRINT ONLY |
| `full_chain.py` | **the whole thing live**: camera -> supercombo -> curvature -> modelV2 bus -> controlsd -> steering torque |
| `inspect_action.py` | probe supercombo step() output structure |

## Key facts

- supercombo `step()` returns: path_xyz(33,3), path_vel, lane_lines(4,33,2),
  road_edges(2,33,2), lane_prob(8), lead_prob(3), pose(6), hidden_state(512).
  **NO curvature/action field** — curvature is derived (curvature_lib.py).
- controlsd subscribes to 14 topics, publishes carControl/controlsState.
  Mazda uses `LatControlTorque` (torque control, not angle/curvature).
- fork patches: `cereal/__init__.py` exports `car` (fixes capnp duplicate);
  `controlsd.py` curvature -> curvatureDEPRECATED (schema version skew).

## What is NOT here

No transmit/actuation path. Every script computes or reads; nothing sends a
steering command to the car's EPS. On-car actuation is the staged in-person
bring-up (harness -> fingerprint -> dashcam -> engaged) with a relay + the
panda safety limits live — not any script in this directory.
