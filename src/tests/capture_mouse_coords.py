"""
capture_mouse_coords.py — Show mouse position relative to the AC window in real-time.

Usage:
    .\\AssetoCorsa\\Scripts\\python.exe tests/capture_mouse_coords.py

Instructions:
    1. Make sure Assetto Corsa is running and visible.
    2. Run this script — it will print the AC window bounds.
    3. Hover your mouse over the element you want to click (e.g. Drive icon).
    4. Press Ctrl+C to stop. The last printed coordinates are your click target.
"""

import ctypes
import time

user32 = ctypes.windll.user32

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


def find_ac_window():
    for title in ["Assetto Corsa", "AC2"]:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd, title
    return 0, None


def main():
    hwnd, title = find_ac_window()
    if not hwnd:
        print("[ERROR] Assetto Corsa window not found. Is AC running?")
        return

    rect = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    win_w = rect.right - rect.left
    win_h = rect.bottom - rect.top

    print(f"[INFO] Found AC window: '{title}' hwnd={hwnd:#010x}")
    print(f"[INFO] Window bounds: ({rect.left}, {rect.top}) → ({rect.right}, {rect.bottom})")
    print(f"[INFO] Window size: {win_w} x {win_h}")
    print()
    print("Move your mouse over the target element in AC.")
    print("Coordinates shown as: abs(x, y)  |  rel(x, y)  |  frac(x%, y%)")
    print("Press Ctrl+C to stop.\n")

    last = (-1, -1)
    try:
        while True:
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            if (pt.x, pt.y) != last:
                rel_x = pt.x - rect.left
                rel_y = pt.y - rect.top
                frac_x = rel_x / win_w if win_w else 0
                frac_y = rel_y / win_h if win_h else 0
                print(
                    f"  abs=({pt.x:5d}, {pt.y:5d})  "
                    f"rel=({rel_x:5d}, {rel_y:5d})  "
                    f"frac=({frac_x:.3f}, {frac_y:.3f})"
                )
                last = (pt.x, pt.y)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n[DONE] Last position — rel=({rel_x}, {rel_y})  frac=({frac_x:.3f}, {frac_y:.3f})")
        print(f"\nTo use in test_ac_lifecycle.py:")
        print(f"    click_x = rect.left + int(win_w * {frac_x:.3f})")
        print(f"    click_y = rect.top  + int(win_h * {frac_y:.3f})")


if __name__ == "__main__":
    main()
