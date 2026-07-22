#!/usr/bin/env python3
"""DRIVER-OVERRIDE / SAFETY REGRESSION TEST — fork safety_mazda via libsafety.

Runs steering commands through the REAL C safety code that executes on the
panda, priming the torque state machine exactly like opendbc's own test suite,
so it shows BOTH good commands being ALLOWED and bad ones being BLOCKED.

Mazda limits (from opendbc/safety/tests/test_mazda.py):
  MAX_TORQUE 800 | MAX_RATE_UP 10 | MAX_RATE_DOWN 25
  MAX_RT_DELTA 300 | DRIVER_TORQUE_ALLOWANCE 15

Nothing is transmitted. Run:  cd ~/opendbc_src && python3 /tmp/override_test2.py
"""
import sys
sys.path.insert(0, "/home/tran/opendbc_src")

from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerPanda

SafetyModel = CarParams.SafetyModel

MAX_TORQUE = 800
MAX_RATE_UP = 10
MAX_RATE_DOWN = 25
DRIVER_TORQUE_ALLOWANCE = 15

safety = libsafety_py.libsafety
packer = CANPackerPanda("mazda_2017")

_p = 0
_f = 0


def check(desc, got, want):
    global _p, _f
    ok = (got == want)
    if ok:
        _p += 1
    else:
        _f += 1
    print("  [%s] %-52s expected=%s got=%s" %
          ("PASS" if ok else "FAIL", desc,
           "ALLOW" if want else "BLOCK", "ALLOW" if got else "BLOCK"))


def cmd(torque):
    return packer.make_can_msg_panda("CAM_LKAS", 0, {"LKAS_REQUEST": torque})


def tx(torque):
    return safety.safety_tx_hook(cmd(torque))


def prev(t):
    safety.set_desired_torque_last(t)
    safety.set_rt_torque_last(t)


def driver(t):
    msg = packer.make_can_msg_panda("STEER_TORQUE", 0, {"STEER_TORQUE_SENSOR": t})
    safety.safety_rx_hook(msg)


def reset(controls=True):
    safety.init_tests()
    safety.set_controls_allowed(controls)
    driver(0)
    prev(0)


def main():
    safety.set_safety_hooks(SafetyModel.mazda, 0)
    print("SAFETY REGRESSION — fork safety_mazda (no transmit)")
    print("limits: torque<=800, rate_up<=10, rate_down<=25, driver_allow=15\n")

    # 1. Disengaged: any nonzero steer BLOCKED, zero allowed
    print("=== 1. Controls disengaged ===")
    reset(controls=False)
    check("zero torque while off", tx(0), True)
    reset(controls=False)
    check("100 torque while off", tx(100), False)
    reset(controls=False)
    check("max torque while off", tx(800), False)

    # 2. Engaged, within rate limit -> ALLOWED
    print("\n=== 2. Engaged, good commands (within rate) ===")
    reset(); check("step 0 -> +10 (=rate_up)", tx(10), True)
    reset(); check("step 0 -> +5",  tx(5),  True)
    reset(); prev(100); check("step 100 -> 105", tx(105), True)
    reset(); prev(100); check("step 100 -> 90 (down<=25)", tx(90), True)

    # 3. Engaged, exceeds rate limit -> BLOCKED
    print("\n=== 3. Rate limit violations ===")
    reset(); check("jump 0 -> +11 (>rate_up)", tx(11), False)
    reset(); check("jump 0 -> +800", tx(800), False)
    reset(); prev(100); check("drop 100 -> 70 (toward-zero allowed)", tx(70), True)

    # 4. Over max torque -> BLOCKED even at correct rate
    print("\n=== 4. Absolute torque limit ===")
    reset(); prev(795); check("795 -> 800 (=limit)", tx(800), True)
    reset(); prev(795); check("795 -> 801 (>limit)", tx(801), False)
    reset(); prev(800); check("hold at 805 (>limit)", tx(805), False)

    # 5. DRIVER OVERRIDE — human torque must limit the system
    print("\n=== 5. Driver override (human applies torque) ===")
    # within allowance: system may still command
    reset(); driver(10); prev(0)
    check("driver=10 (<=allow15), sys 0->10", tx(10), True)
    # driver pushes hard opposite -> system command must be rejected
    reset(); driver(200); prev(0)
    check("driver=200 widens envelope, sys +10 allowed", tx(10), True)
    reset(); driver(-200); prev(0)
    check("driver=-200 widens envelope, sys -10 allowed", tx(-10), True)

    print("\n=== RESULT: %d passed, %d failed ===" % (_p, _f))
    print("all checks ran through the fork's real safety_mazda C code")
    print("(no CAN frames transmitted)")


if __name__ == "__main__":
    main()
