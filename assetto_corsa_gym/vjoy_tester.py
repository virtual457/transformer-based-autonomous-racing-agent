"""
vJoy Tester — Tests vJoy device directly without needing AC running.
Sends axis values and confirms vJoy is working.

Usage:
    python vjoy_tester.py

No AC needed. Just checks vJoy device is responding.
"""

import sys
import os
import time

sys.path.extend([os.path.abspath('./assetto_corsa_gym')])

from AssettoCorsaPlugin.plugins.sensors_par.vjoy import vJoy as VJoy

def main():
    print("Initializing vJoy Device 1...")
    try:
        vjoy = VJoy()
        if not vjoy.open():
            print("✗ Could not acquire vJoy Device 1. Is another app holding it?")
            return
        print("✓ vJoy Device 1 acquired\n")
    except Exception as e:
        print(f"✗ Failed to connect to vJoy: {e}")
        print("  Make sure vJoy is installed and Device 1 is enabled in vJoy Config.")
        return

    def send(steer, acc, brake):
        # steer: [-1,1] → wAxisX [0, 32768]
        # acc:   [0,1]  → wAxisY [0, 32768]
        # brake: [0,1]  → wAxisZ [0, 32768]
        x = int((steer + 1) * 16384)
        y = int(acc   * 32768)
        z = int(brake * 32768)
        pos = vjoy.generateJoystickPosition(wAxisX=x, wAxisY=y, wAxisZ=z)
        vjoy.update(pos)
        return x, y, z

    print("Sending test inputs — open joy.cpl to watch axes move in real-time.")
    print("Win+R → joy.cpl → vJoy Device → Properties\n")

    tests = [
        ("Full throttle  (acc=1)",       0.0,  1.0,  0.0),
        ("Full brake     (brake=1)",      0.0,  0.0,  1.0),
        ("Steer full left  (steer=-1)",  -1.0,  0.0,  0.0),
        ("Steer full right (steer=+1)",   1.0,  0.0,  0.0),
        ("Center / release",              0.0,  0.0,  0.0),
    ]

    try:
        for label, steer, acc, brake in tests:
            x, y, z = send(steer, acc, brake)
            print(f"  {label:<35}  X={x:5d}  Y={y:5d}  Z={z:5d}")
            time.sleep(1.5)

        print("\n✓ All test inputs sent. If axes moved in joy.cpl, vJoy is working correctly.")

    except KeyboardInterrupt:
        pass
    finally:
        send(0.0, 0.0, 0.0)
        vjoy.close()
        print("Axes reset. vJoy released.")

if __name__ == "__main__":
    main()
