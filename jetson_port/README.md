# openpilot on Jetson Orin + DIY STM32F446 panda → Mazda CX-5 2023

Port of comma.ai openpilot to run on an **NVIDIA Jetson Orin Nano** with a **DIY panda**
(STM32F446RE Nucleo + 2 CAN transceivers), targeting a **Mazda CX-5 2023** (`MAZDA_CX5_2022`
platform, supported in opendbc).

## Status

| Component | State |
|-----------|-------|
| supercombo driving model on Jetson (TensorRT) | ✅ working, 31 Hz temporal (>20 Hz req) |
| camera → model warp (calibrated) | ✅ ported + verified |
| output decode (path/lanes/lead/pose) | ✅ |
| recurrent feature feedback (temporal) | ✅ |
| panda F446 firmware (serial link) | ✅ written — untested, needs board |
| pandad serial handle | ✅ written — untested |
| SConscript build / compile | ⬜ TODO |
| bench CAN loopback | ⬜ TODO (before car!) |
| on-car | ⬜ TODO (safety-critical) |

## `model/` — the driving model on the Jetson (no openpilot install needed)

- `op_frame.py` — camera → model input. Ports openpilot's `get_warp_matrix` (calibration
  perspective warp) + YUV420 → 12-channel packing. IMX477 intrinsics `fx≈W*0.7` (tune).
- `op_parse.py` — decodes the 2576-float output using the ONNX-embedded `output_slices` metadata.
- `op_stream.py` — **`SupercomboRunner`**: full streaming runner with recurrent feedback
  (feat_q depth 96, subsample /4 → features_buffer 24×512). `.step(bgr)` → driving plan.
- `op_pipeline.py`, `op_run.py` — single-shot test harnesses.

Build the engine on the Orin:
```
trtexec --onnx=driving_supercombo.onnx --saveEngine=supercombo_fp32.trt --skipInference
```
FP32 runs at 42 Hz. (fp16 blocked by a TRT-10.16 gelu-fusion bug; FP32 is fast enough.)

## `panda_f446_firmware/` — DIY panda on the Nucleo-F446RE

Faithful port of comma's last STM32F4 panda firmware (panda commit `3dc21386`, before "bye bye f4")
to the F446, talking to the Jetson over the **ST-Link Virtual COM Port (serial)** instead of USB.

- `drivers/serial_comms.h` — UART transport, a faithful clone of the SPI transport framing.
  Reuses panda's transport-agnostic `comms_control_handler` / `comms_can_read` / `comms_can_write`.
- `drivers/serial_uart_raw.h` — blocking USART2 byte I/O.
- `stm32f446/peripherals.h` — pin map: **USART2 PA2/PA3** (VCP), **CAN1 PB8/PB9**, **CAN2 PB5/PB6** (all AF9).
- `boards/nucleo.h` — `board_nucleo` definition (2 CAN, minimal).
- `stm32f446/clock.h` — 180 MHz + PLLSAI 48 MHz (F446 dual-PLL).
- `stm32f446/stm32f446_flash.ld` — 512 K flash / 128 K RAM.
- `stm32f446/inc/stm32f446xx.h`, `startup_stm32f446xx.s` — official ST CMSIS.

**Wiring:** transceiver 1 → CAN1 (PB8 RX, PB9 TX) → car main bus; transceiver 2 → CAN2 (PB5 RX,
PB6 TX) → car camera/LKAS bus. Link + power + flash all over the Nucleo's single ST-Link USB.

## `pandad_serial/` — Jetson side

- `serial.cc` — `PandaSerialHandle`: same interface as `PandaSpiHandle`, over `/dev/ttyACM`.
- `panda_comms_serial.h` — class decl to merge into `selfdrive/pandad/panda_comms.h`.

## Next steps (in order — SAFETY FIRST)

1. SConscript entry to compile `board_nucleo`; build firmware with `arm-none-eabi-gcc`.
2. Flash via ST-Link; verify the F446 enumerates and pandad connects over serial.
3. **Bench CAN loopback** — CAN1↔CAN2 wired together, verify send/recv, BEFORE any car.
4. Keep comma's Mazda safety model (`safety_mazda`) fully intact. Never bypass safety on-car.
5. Only then: car harness, read-only CAN first, then actuation with extreme care.

## ⚠️ Safety

This ultimately controls a real car's steering/gas/brake. comma's safety model exists for a reason.
Bench-test everything. Read-only before actuation. Never rush the on-car step.
