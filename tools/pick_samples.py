#!/usr/bin/env python3
"""Pick varied SP samples for ML/RB comparison.

Reads BMS.Tools .attrs.json snapshots, matches to available packages in
BMS.Generator/Insane BMS, and picks samples spanning different axis profiles.
"""
import os
import json
import re

ATTRS_DIR = r"C:/Repos/BMS.Tools/samples/_attrs"
SAMPLES_DIR = r"C:/Repos/BMS.Tools/samples"
PKG_DIR = r"C:/Repos/BMS.Generator/Insane BMS (2025-12-14)"

available_pkgs = set(os.listdir(PKG_DIR))
available_pkgs_clean = {p.lower().replace(" ", "").replace("　", ""): p for p in available_pkgs}


def find_pkg(title_full):
    title_clean = title_full.lower().replace(" ", "").replace("　", "")
    for pkg_clean, pkg in available_pkgs_clean.items():
        if title_clean[:25] in pkg_clean or pkg_clean[:25] in title_clean:
            return pkg
    return None


# Load all SP samples + their axis values
records = []
for f in os.listdir(ATTRS_DIR):
    if not f.startswith("SP__"):
        continue
    base = f[:-11]
    parts = base.split("___")
    if len(parts) < 2:
        continue
    title_part = parts[0]
    chart_part = parts[1] if len(parts) >= 2 else ""
    m = re.match(r"^SP__([^_]+)__(.+)$", title_part)
    if not m:
        continue
    cat, title_full = m.group(1), m.group(2).strip()

    pkg = find_pkg(title_full)
    if not pkg:
        continue

    try:
        with open(os.path.join(ATTRS_DIR, f), encoding="utf-8") as fp:
            d = json.load(fp)
    except Exception:
        continue

    # Confirm it's actually SP
    if d.get("mode") != "SP":
        continue

    # Pull x_* axes
    axes = {k: d.get(k) for k in (
        "x_chord", "x_stream", "x_scratch", "x_soft",
        "x_ln", "x_stair", "x_peak", "x_distraction", "x_jack"
    )}
    headers = d.get("headers", {})

    # The BMS.Tools sample filename has format SP__cat__title___<chart_stem>.<ext>
    # The corresponding chart file inside the package has filename <chart_stem>.<ext>
    # (or close variation). Search for it.
    bms_ext = os.path.splitext(d.get("source", "") or "")[1] or ".bms"
    chart_stem = chart_part.rstrip(".")  # e.g. "undertheden_sabunnnnnnnn"
    pkg_path = os.path.join(PKG_DIR, pkg)
    bms_filename = None
    if os.path.isdir(pkg_path):
        for fn in os.listdir(pkg_path):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in (".bms", ".bme", ".bml"):
                continue
            if stem == chart_stem or fn == f"{chart_stem}{bms_ext}":
                bms_filename = fn
                break
    bms_exists = bms_filename is not None

    records.append({
        "attrs_file": f,
        "title": title_full,
        "chart": chart_part,
        "category": cat,
        "package": pkg,
        "bms_filename": bms_filename,
        "bms_exists": bms_exists,
        "headers": headers,
        "axes": axes,
    })


print(f"Total matched SP records: {len(records)}")
print(f"With BMS file present in package: {sum(1 for r in records if r['bms_exists'])}")

# Filter to only those with bms file present
records = [r for r in records if r["bms_exists"]]
print(f"Usable: {len(records)}")
print()

# For each axis, pick top sample where that axis dominates
def domain_score(r, dom_axis):
    """Higher score for samples where dom_axis is high but others moderate."""
    axes = r["axes"]
    dom_val = axes.get(dom_axis, 0) or 0
    other_axes = ["x_chord", "x_stream", "x_scratch", "x_peak", "x_stair", "x_ln"]
    other_axes = [a for a in other_axes if a != dom_axis]
    other_max = max((axes.get(a, 0) or 0) for a in other_axes)
    # Want dom high (>0.5) but others not all high
    return dom_val - 0.5 * other_max


targets = [
    ("x_stream", "stream-heavy"),
    ("x_chord", "chord-heavy"),
    ("x_peak", "burst/peak"),
    ("x_stair", "stair pattern"),
    ("x_scratch", "scratch heavy"),
    ("x_distraction", "distraction"),
]

print("=== Picks by axis ===\n")
for axis, label in targets:
    sorted_by_dom = sorted(records, key=lambda r: -domain_score(r, axis))
    # Pick top with dom_val > 0.5
    picks = [r for r in sorted_by_dom if (r["axes"].get(axis, 0) or 0) > 0.5][:3]
    print(f"\n--- {axis} ({label}) ---")
    for r in picks:
        ax = r["axes"]
        ax_str = " ".join(f"{k[2:]}={v:.2f}" if isinstance(v, (int, float)) else f"{k[2:]}=N/A"
                          for k, v in ax.items() if v is not None and not isinstance(v, str))
        title = r["title"][:50]
        chart = r["chart"][:40]
        pkg = r["package"][:50]
        print(f"  {title}")
        print(f"    pkg: {pkg}")
        print(f"    bms: {r['bms_filename']}")
        print(f"    axes: {ax_str}")

# Also pick a balanced/moderate one
print("\n--- balanced (0.3 < all axes < 0.6) ---")
balanced = [r for r in records if all(
    0.25 < (r["axes"].get(a, 0) or 0) < 0.6
    for a in ("x_chord", "x_stream", "x_peak", "x_stair")
)]
for r in balanced[:3]:
    ax = r["axes"]
    ax_str = " ".join(f"{k[2:]}={v:.2f}" if isinstance(v, (int, float)) else f"{k[2:]}=N/A"
                      for k, v in ax.items() if v is not None and not isinstance(v, str))
    print(f"  {r['title'][:50]} | pkg: {r['package'][:50]}")
    print(f"    bms: {r['bms_filename']}")
    print(f"    axes: {ax_str}")
