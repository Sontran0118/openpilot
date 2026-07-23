# Jetson Mazda port baseline — 2026-07-22

This records the last verified prototype state before the deterministic-build
and CAN-topology corrections. It is a preservation baseline, not an
actuation-ready release.

## Source revisions

| tree | revision |
|---|---|
| `~/op_fork` | `90676e683d66e621cc7bba45f12a6d75b0c2b871` |
| GitHub `origin/jetson-mazda-port` before preservation | `d23e3a6faacb98b61aec045fe7edd78f0cbdfe0c` |
| `~/opendbc_src` | `8ddffb37c8d8ea51727042ebe8743014ef5b5bfc` |
| `~/panda_base` / `~/panda_f446` base | `3dc21386239e3073a623156b75901aa302340d6c` |
| `~/msgq_build` | `425b61a60358895b894e9a6920b1d3e06903132e` |

The openpilot branch was three commits ahead of GitHub at audit start:
`c4f5d15`, `c807270`, and `90676e6`.

## Platform and toolchain

- Jetson: Linux `6.8.12-1021-tegra`, aarch64
- L4T: R39 revision 2.0, GCID 45755727
- Python 3.12.3
- GCC/G++ 13.3.0
- arm-none-eabi-gcc 13.2.1 (Ubuntu package `15:13.2.rel1-2`)
- Cap'n Proto 1.0.1
- TensorRT 10.16.2 (`trtexec` v101602)
- Cython 3.2.8, cffi 2.1.0, crcmod 1.7, pyzmq 27.1.0
- setproctitle 1.3.7, tinygrad 0.13.0, pyserial 3.5, pycapnp 2.1.0

## Preserved artifact hashes

```text
3a660fcc14342b60da0d998043de986522a4f726ebb0e76bae9318a148b90ff4  bootstub.panda_f446.bin
45b44a0c992ace18375db351c57614969e4be60c26527e2d93865ff094e1520f  panda_f446.bin.signed
103b0c10032f4a3c6639572b40f68b40ff41f64e15147b111d1934a48d5870c1  supercombo_fp32.trt
b447d4fd0329aab266fe2903ed3b565fe601808fe280eb26f55b52f509a7778d  params_pyx.so
3018c7ad8e66fa2358a473d01b14fa28d806ea319d0fdfa5b3e6fb69aeed06e5  ipc_pyx.so
```

## Verified at this baseline

- STM32 serial health response: 58 bytes.
- Vehicle-state response: 27 bytes; live Mazda traffic decoded on the MCU.
- Cereal/msgq publish/subscribe round trip passed.
- Mazda safety regression passed 16/16.
- Multi-process synthetic controlsd test returned carControl 3/3.
- The panda reconstruction patch applies cleanly to panda `3dc21386`.

## Known blockers and accuracy notes

- A clean firmware build currently fails because `vehicle_state_update()` is
  compiled into the bootstub and is unused under `-Werror`. The preserved
  bootstub binary predates that source change and must be treated as stale.
- F446 CAN3 is currently aliased to CAN2 while `PANDA_CAN_CNT` remains three.
  This duplicates peripheral initialization and IRQ registration and must be
  replaced by an explicit physical CAN1/logical bus 0 and physical
  CAN2/logical bus 2 mapping.
- Wheel speeds in the 0xd5 vehicle-state packet are raw DBC values; the
  documented `kph*100` contract omits the Mazda `-100 km/h` offset.
- The demonstrated control chain ends at `carControl`. Mazda CarController,
  mazdacan, pandad, and panda safety have not yet been joined into one send
  path.
- No vehicle actuation is authorized by this baseline. Hardware output remains
  gated until the bench, replay, relay, and in-person safety stages pass.
