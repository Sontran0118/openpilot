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
| panda F446 firmware (serial link) | ✅ **builds, flashes, RUNS on hardware** |
| pandad serial handle | ✅ **verified against the real board** |
| SConscript build / compile | ✅ `board/obj/panda_f446.bin.signed` |
| bench CAN loopback | ✅ **PASS** — both buses loop back |
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

## Building + flashing the F446 firmware (VERIFIED 2026-07-21)

The port files here are *overlays* on comma's panda at commit `3dc21386` (the last F4 commit,
before "bye bye f4"). Reproduce the working tree:

```bash
# toolchain
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi libnewlib-arm-none-eabi \
                 stlink-tools gdb-multiarch
pip install --break-system-packages scons pycryptodome

# base trees
git clone https://github.com/commaai/panda.git panda_base && \
  (cd panda_base && git checkout 3dc21386)
git clone https://github.com/commaai/opendbc.git opendbc_src && \
  (cd opendbc_src && git checkout 8ddffb37)   # version-matched; pip opendbc is TOO NEW

# assemble
cp -r panda_base panda_f446
cd panda_f446
cp -r  <this>/panda_f446_firmware/stm32f446        board/
cp     <this>/panda_f446_firmware/drivers/serial_*.h  board/drivers/
cp     <this>/panda_f446_firmware/boards/nucleo.h     board/boards/
git apply <this>/panda_base_patches/panda_base_3dc21386.patch

# build + flash
scons -j2 board/obj/bootstub.panda_f446.bin board/obj/panda_f446.bin.signed
sudo st-flash --reset write board/obj/bootstub.panda_f446.bin 0x8000000
sudo st-flash --reset write board/obj/panda_f446.bin.signed  0x8004000

# MUST PASS before any car connection
sudo python3 bench_can_loopback.py /dev/ttyACM0
```

### Wiring (Nucleo-F446RE)

The board silkscreens **Arduino** names, not port names:

| Signal | Port | Board label |
|--------|------|-------------|
| CAN1 RX | PB8 | **D15** (SCL) |
| CAN1 TX | PB9 | **D14** (SDA) |
| CAN2 RX | PB6 | **D10** |
| CAN2 TX | PB5 | **D4** |
| 3.3V / GND / VIN | — | **CN6** (left header) |

Transceivers: SN65HVD230 (3.3V native). **Remove the 120Ω termination jumper on the car side** —
the vehicle bus is already terminated at both ends. Link + power + flashing all run over the single
mini-USB (ST-Link VCP, `/dev/ttyACM0`), panda protocol at 1.5 Mbaud on USART2 (PA2/PA3).

### Port gotchas worth knowing

- `MCU_IDCODE` must be **`0x421`** (F446), not the F413's `0x463` — otherwise `early_initialization()`
  takes the "wrong chip" path and HardFaults in `led_init()` before GPIO clocks are on.
- The F446 has **no CAN3**: CAN3 IRQ branches removed, `CAN3` aliased to `CAN2` in `bxcan.h`.
- **128 K RAM** (F413 had 256 K): `CAN_RX_BUFFER_SIZE 4096→1024`, `CAN_TX_BUFFER_SIZE 416→128`,
  `REGISTER_MAP_SIZE 0x3FF→0xFF`. That last one is used as a **bitmask**, so it must stay `2^n-1`.
- The stock `main()` runs a multi-second LED fade between comms calls. Harmless for USB pandas
  (interrupt-driven) but it **starves a polled UART** — exactly one transaction worked per reset
  until the fade was skipped for `PANDA_NUCLEO`.
- Clock is 180 MHz / APB1 45 / APB2 90; `stm32f4_config.h` must agree with `clock.h` or every
  baud rate and timer is wrong.

### Debugging on-target

```bash
sudo st-util -n &
gdb-multiarch -q -ex "target extended-remote localhost:4242" \
  -ex bt board/obj/panda_f446/main.elf
```
`LR=0xFFFFFFF9` with `xPSR & 0x1FF == 3` means you are sitting in HardFault.

## Next steps (in order — SAFETY FIRST)

1. SConscript entry to compile `board_nucleo`; build firmware with `arm-none-eabi-gcc`.
2. Flash via ST-Link; verify the F446 enumerates and pandad connects over serial.
3. **Bench CAN loopback** — CAN1↔CAN2 wired together, verify send/recv, BEFORE any car.
4. Keep comma's Mazda safety model (`safety_mazda`) fully intact. Never bypass safety on-car.
5. Only then: car harness, read-only CAN first, then actuation with extreme care.

## ⚠️ Safety

This ultimately controls a real car's steering/gas/brake. comma's safety model exists for a reason.
Bench-test everything. Read-only before actuation. Never rush the on-car step.
