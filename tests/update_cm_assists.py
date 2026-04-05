"""
update_cm_assists.py — Directly patch CM's Values.data binary to set assist values.

Background
----------
Content Manager stores its persistent settings in:
    C:\\Users\\<user>\\AppData\\Local\\AcTools Content Manager\\Values.data

Format:
    - First byte: 0x0B  (LZF flag)
    - Remaining bytes: LZF-compressed UTF-8 text

After decompression the text is a simple flat key/value store:
    version: 2\n
    key1\tvalue1\n
    key2\tvalue2\n
    ...

The key __QuickDrive_Main holds a JSON object. String values that are themselves
JSON objects (ModeData, AssistsData, TrackPropertiesData) are stored with CM's
double-backslash escaping: each " in the nested JSON is preceded by two backslashes
(literal chars 0x5C 0x5C 0x22) rather than the standard JSON \\".

This encoding is a Newtonsoft.Json serialisation artefact. We must handle it on
both read and write.

This script:
    1. Reads Values.data
    2. LZF-decompresses the payload (skipping the 0x0B header byte)
    3. Finds the __QuickDrive_Main key
    4. Parses the outer JSON (fixing double-backslash encoding first)
    5. Updates AssistsData with the required values
    6. Sets asc: true (AssistsChanged flag — tells CM to apply the assists on next launch)
    7. Re-encodes with double-backslash escaping for nested JSON
    8. LZF-compresses the modified text
    9. Writes back with the 0x0B header byte

CM MUST NOT be running when this script is called — the file is memory-mapped
by CM while it runs, and writes will be silently lost or cause corruption.

Usage
-----
    .\\AssetoCorsa\\Scripts\\python.exe tests/update_cm_assists.py [--dry-run]

    --dry-run   Print what would be changed without writing to disk.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ── Target file ───────────────────────────────────────────────────────────────

VALUES_DATA_PATH = Path(
    r"C:\Users\chand\AppData\Local\AcTools Content Manager\Values.data"
)

# ── Required assist values ────────────────────────────────────────────────────

REQUIRED_ASSISTS = {
    "AutoShifter":       True,
    "AutoClutch":        True,
    "Abs":               2,
    "TractionControl":   2,
    "StabilityControl":  100.0,
    "Damage":            0.0,
    "TyreWear":          0.0,
    "FuelConsumption":   0.0,
    "IdealLine":         True,
    "AutoBlip":          True,
    "AutoBrake":         False,
    "VisualDamage":      False,
    "TyreBlankets":      False,
    "SlipSteam":         1.0,
}


# ── Pure-Python LZF decompressor ──────────────────────────────────────────────
#
# LZF is a byte-stream format with two token types:
#
#   Literal run  (ctrl < 32):
#       ctrl encodes (literal_count - 1).  Read ctrl+1 raw bytes and copy to output.
#
#   Back-reference (ctrl >= 32):
#       length = (ctrl >> 5) - 1
#       If length == 6: read one more byte and add to length  (extended-length form)
#       offset_hi = (ctrl & 0x1F) << 8
#       offset_lo = next byte
#       distance  = offset_hi | offset_lo + 1
#       Copy (length + 3) bytes from (out_pos - distance) to output.
#       The copy may overlap (run-length expansion); must copy byte-by-byte.
#
# Reference: Marc Lehmann's liblzf source; CM uses the same byte encoding in
# AcTools.LapTimes / CustomCombo / Helpers/LZF.cs.

def lzf_decompress(data: bytes) -> bytes:
    """Decompress raw LZF bytes (without the 0x0B CM header byte)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            # Literal run: ctrl+1 bytes follow
            count = ctrl + 1
            out += data[i : i + count]
            i += count
        else:
            # Back-reference
            length = (ctrl >> 5) - 1
            if length == 6:                # extended length
                length += data[i]
                i += 1
            offset = ((ctrl & 0x1F) << 8) | data[i]
            i += 1
            ref = len(out) - offset - 1
            length += 3                    # minimum match length is 3
            for _ in range(length):
                out.append(out[ref])
                ref += 1
    return bytes(out)


# ── Pure-Python LZF compressor ────────────────────────────────────────────────
#
# This is a simple "literals-only" LZF compressor: it emits every byte as a
# literal run (no back-references).  The resulting file is larger than an
# optimally-compressed LZF stream but is 100% valid and CM decompresses it
# correctly.  Since Values.data is ~4–7 KB total, the size penalty is negligible.
#
# Literal-run format: one control byte (value = count-1, max 31) followed by
# count literal bytes (1 ≤ count ≤ 32).

def lzf_compress_literals_only(data: bytes) -> bytes:
    """Compress bytes using LZF literal-only encoding (no back-references)."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        # Emit up to 32 literal bytes per run
        count = min(32, n - i)
        out.append(count - 1)          # control byte: literal run
        out += data[i : i + count]
        i += count
    return bytes(out)


# ── CM double-backslash escaping helpers ──────────────────────────────────────
#
# CM's Newtonsoft.Json serialiser stores nested-JSON strings with every " in the
# nested JSON preceded by two literal backslashes (0x5C 0x5C 0x22) instead of
# the standard single-escape 0x5C 0x22.
#
# Example (bytes in the decompressed file):
#   "AssistsData":"{\\\"IdealLine\\\":false, ...}"
#   (each " in the nested JSON is preceded by two backslash chars)
#
# To parse: replace every occurrence of two-backslash-then-quote (\\\") with
# just a quote (") so that the outer JSON becomes well-formed.
#
# To re-encode: take the inner JSON string and replace every " with two
# backslashes + quote (\\\"), then embed as a normal JSON string value.

_DOUBLE_BS_QUOTE = "\\\\\""       # the pattern: backslash backslash quote (3 chars)
_SINGLE_QUOTE    = "\""           # a plain quote (1 char)


def _cm_unescape_nested(val_str: str) -> str:
    """Replace CM's \\\" → \" in the outer JSON string to make it parseable."""
    return val_str.replace(_DOUBLE_BS_QUOTE, _SINGLE_QUOTE)


def _cm_escape_nested(inner_json: str) -> str:
    """Replace \" → \\\" to restore CM's double-backslash encoding."""
    return inner_json.replace(_SINGLE_QUOTE, _DOUBLE_BS_QUOTE)


# ── Core update logic ─────────────────────────────────────────────────────────

def update_assists_in_values_data(
    path: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """
    Read Values.data, update AssistsData in __QuickDrive_Main, write back.

    Returns a dict with:
        'changed': list of (key, old_value, new_value) tuples
        'unchanged': list of key names whose values were already correct
    """
    if not path.exists():
        raise FileNotFoundError(f"Values.data not found at: {path}")

    raw = path.read_bytes()

    if not raw or raw[0] != 0x0B:
        raise ValueError(
            f"Values.data does not start with 0x0B (got {hex(raw[0]) if raw else 'empty'}). "
            "File format may have changed."
        )

    # ── Decompress ────────────────────────────────────────────────────────────
    compressed_payload = raw[1:]
    decompressed = lzf_decompress(compressed_payload)
    text = decompressed.decode("utf-8")

    # ── Parse key/value store ────────────────────────────────────────────────
    lines = text.split("\n")
    target_key = "__QuickDrive_Main"
    target_line_idx = None
    for idx, line in enumerate(lines):
        if line.startswith(target_key + "\t"):
            target_line_idx = idx
            break

    if target_line_idx is None:
        raise KeyError(
            f"Key '{target_key}' not found in Values.data. "
            "Open CM once to create a Quick Drive session, then re-run this script."
        )

    val_str = lines[target_line_idx][len(target_key) + 1:]  # skip "key\t"

    # ── Parse outer JSON (fix CM's double-backslash escaping first) ───────────
    parseable = _cm_unescape_nested(val_str)
    outer = json.loads(parseable)

    # ── Parse inner AssistsData JSON ─────────────────────────────────────────
    assists_raw = outer.get("AssistsData", "")
    if not assists_raw:
        # AssistsData missing — start with an empty dict
        assists = {}
    else:
        assists = json.loads(assists_raw)

    # ── Apply required values and collect changes ─────────────────────────────
    changed   = []
    unchanged = []
    for key, new_val in REQUIRED_ASSISTS.items():
        old_val = assists.get(key)
        if old_val != new_val:
            changed.append((key, old_val, new_val))
            assists[key] = new_val
        else:
            unchanged.append(key)

    # Also check asc flag (AssistsChanged — tells CM to re-apply on next launch)
    old_asc = outer.get("asc")
    if old_asc is not True:
        changed.append(("asc (top-level)", old_asc, True))
    outer["asc"] = True

    if not changed:
        print("[update_cm_assists] No changes needed — all assist values already correct.")
        return {"changed": [], "unchanged": unchanged}

    if dry_run:
        print("[update_cm_assists] DRY RUN — changes that would be applied:")
        for key, old, new in changed:
            print(f"  {key}: {old!r} -> {new!r}")
        return {"changed": changed, "unchanged": unchanged}

    # ── Re-encode AssistsData with CM's double-backslash escaping ─────────────
    new_assists_json = json.dumps(assists, separators=(",", ":"))
    outer["AssistsData"] = _cm_escape_nested(new_assists_json)

    # ── Re-serialise outer JSON ───────────────────────────────────────────────
    # json.dumps will escape the backslashes in AssistsData correctly because
    # _cm_escape_nested produced a Python str containing literal backslashes,
    # and json.dumps will encode those as \\ in the JSON output — recreating
    # the 0x5C 0x5C 0x22 byte pattern CM expects.
    new_val_str = json.dumps(outer, separators=(",", ":"), ensure_ascii=False)

    # ── Rebuild the key/value store text ────────────────────────────────────
    lines[target_line_idx] = target_key + "\t" + new_val_str
    new_text = "\n".join(lines)
    new_decompressed = new_text.encode("utf-8")

    # ── LZF-compress and write back ───────────────────────────────────────────
    new_compressed = lzf_compress_literals_only(new_decompressed)
    new_raw = bytes([0x0B]) + new_compressed

    # Back up the original before overwriting
    backup_path = path.with_suffix(".data.bak")
    shutil.copy2(path, backup_path)

    path.write_bytes(new_raw)

    return {"changed": changed, "unchanged": unchanged}


# ── Internal function for ac_lifecycle integration ────────────────────────────

def _update_cm_assists_values_data(dry_run: bool = False) -> None:
    """
    Patch CM's Values.data to apply the required RL training assists.

    Call this from write_session_config() BEFORE launching CM.
    CM must NOT be running when this is called.

    Prints a summary of what was changed or a no-op message if already correct.
    Raises on file-not-found or format errors (caller should handle or let propagate).
    """
    result = update_assists_in_values_data(VALUES_DATA_PATH, dry_run=dry_run)
    if result["changed"]:
        if not dry_run:
            print("[update_cm_assists] Applied changes to Values.data:")
            for key, old, new in result["changed"]:
                print(f"  {key}: {old!r} -> {new!r}")
            backup = VALUES_DATA_PATH.with_suffix(".data.bak")
            print(f"[update_cm_assists] Backup saved to: {backup}")
    # unchanged values are not printed to keep output concise


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch CM Values.data to set RL training assist values.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print changes without writing to disk.",
    )
    args = parser.parse_args()

    # Safety check: is Content Manager running?
    import subprocess
    result = subprocess.run(
        'tasklist /FI "IMAGENAME eq Content Manager.exe" /NH',
        shell=True,
        capture_output=True,
        text=True,
    )
    if "Content Manager.exe" in result.stdout:
        print(
            "[update_cm_assists] ERROR: Content Manager.exe is currently running.\n"
            "Close CM first, then re-run this script.\n"
            "Values.data was NOT modified.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        _update_cm_assists_values_data(dry_run=args.dry_run)
        if not args.dry_run:
            print("[update_cm_assists] Done. Values.data updated successfully.")
        else:
            print("[update_cm_assists] Dry run complete. No files were modified.")
    except FileNotFoundError as e:
        print(f"[update_cm_assists] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"[update_cm_assists] ERROR parsing Values.data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
