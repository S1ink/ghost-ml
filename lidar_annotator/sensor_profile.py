#!/usr/bin/env python3
"""
sensor_profile.py

Derives a sensor profile (the grid geometry scan_grid.py needs) from a
recording of the lidar, so a new lidar needs no hand-entered numbers.

    python3 sensor_profile.py <bag> [-o sensor.json] [--topic TOPIC] [--name TEXT]

It reads some scans and measures:

    layers          the distinct elevation angles of the points
    cols            the azimuth step between neighbouring points of a layer
    azimuth start   where each layer's samples sit on that step
    scan rate       from the header stamps
    intensity       which point field holds it and its full-scale value

This works for a lidar that spins about its z axis with fixed layer
elevations and a fixed azimuth step, recorded in the sensor's own frame
(points not motion-compensated or transformed). The printed check says how
well the recording fits that model; a poor fit means the profile should not
be used. Not supported: non-repeating scan patterns, and layers with
different azimuth steps (the most common step is used and the others are
reported).

Derive the profile once per project and keep it: labels and models are tied
to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan_grid as sg

INTENSITY_FIELDS = ("intensity", "reflectivity", "i", "Intensity", "reflector")


def _layers(elev: np.ndarray) -> np.ndarray:
    """Cluster centres of 1D elevation samples (degrees), ascending."""
    e = np.sort(elev)
    gaps = np.diff(e)
    order = np.argsort(-gaps)[:2048]
    g = np.maximum(gaps[order], 1e-4)
    if len(g) < 3:
        raise ValueError("too few points to find layers")
    # Gaps between layers are far larger than gaps inside one: cut at the
    # biggest relative drop in the sorted gap sizes.
    ratio = g[:-1] / g[1:]
    n_cuts = int(np.argmax(ratio[:1024])) + 1
    cuts = np.sort(order[:n_cuts])
    groups = np.split(e, cuts + 1)
    # Stray points (noise, reflections) form tiny clusters; drop them.
    typical = np.median([len(x) for x in groups])
    groups = [x for x in groups if len(x) >= 0.1 * typical]
    return np.array([np.median(x) for x in groups])


def derive(bag_path: str, topic: str | None = None, name: str | None = None,
           n_scans: int = 40, stride: int = 5, verbose: bool = True) -> dict:
    """Returns a sensor profile dict measured from `bag_path`."""
    say = print if verbose else (lambda *a, **k: None)
    topic = sg.resolve_topic(bag_path, topic)
    scans, stamps, rec, fields, frame = [], [], [], None, ""
    for k, (rec_t, hdr_t, msg, pts) in enumerate(sg.iter_scans(bag_path, topic)):
        stamps.append(hdr_t)
        rec.append(rec_t)
        if fields is None:
            fields, frame = pts.dtype, msg.header.frame_id
            for need in ("x", "y", "z"):
                if need not in fields.names:
                    raise ValueError(f"{topic} has no '{need}' field "
                                     f"(fields: {', '.join(fields.names)})")
        if k % stride == 0 and len(scans) < n_scans:
            scans.append(np.array(pts))
        if len(scans) >= n_scans and k >= n_scans * stride:
            break
    if len(scans) < 3:
        raise ValueError(f"{topic}: only {len(scans)} usable scans")

    def polar(p):
        x, y, z = (p[k].astype(np.float64) for k in ("x", "y", "z"))
        xy = np.hypot(x, y)
        rng = np.hypot(xy, z)
        ok = np.isfinite(rng) & (rng > 0.05)
        return (np.degrees(np.arctan2(z, xy))[ok], np.degrees(np.arctan2(y, x))[ok])

    pol = [polar(p) for p in scans]
    elev_all = np.concatenate([e for e, _ in pol])
    if len(elev_all) < 1000:
        raise ValueError("the sampled scans hold almost no points")
    layers = _layers(elev_all)
    spacing = np.min(np.diff(layers)) if len(layers) > 1 else 1.0

    # Azimuth step of every layer, from neighbouring points within one scan.
    steps = [[] for _ in layers]
    by_layer = [[] for _ in layers]
    spread = []
    for e, az in pol:
        k = np.abs(e[:, None] - layers[None, :]).argmin(axis=1)
        spread.append(np.abs(e - layers[k]))
        for li in range(len(layers)):
            a = np.sort(az[k == li])
            by_layer[li].append(a)
            d = np.diff(a)
            d = d[d > 1e-3]                      # multi-echo: same direction twice
            if len(d) > 8:
                small = d[d < 1.5 * np.percentile(d, 5)]
                steps[li].append(np.median(small))
    layer_step = np.array([np.median(s) if s else np.nan for s in steps])
    layer_cols = np.round(360.0 / layer_step)
    vals, counts = np.unique(layer_cols[np.isfinite(layer_cols)], return_counts=True)
    cols = int(vals[np.argmax(counts)])
    step = 360.0 / cols
    odd = [i for i, c in enumerate(layer_cols) if c != cols]

    # Where each layer's samples sit on the step (circular mean of az mod step).
    az0, jitter = [], []
    for li in range(len(layers)):
        a = np.concatenate(by_layer[li]) if by_layer[li] else np.zeros(0)
        ph = np.exp(2j * np.pi * a / step)
        mean = ph.mean() if len(a) else 1.0
        phase = (np.angle(mean) / (2 * np.pi) * step) % step
        az0.append(round(float(-180.0 + phase), 4))
        res = (a - phase + step / 2) % step - step / 2
        jitter.append(float(np.sqrt(np.mean(res ** 2)) / step) if len(a) else 0.0)

    t = np.array(stamps if len(set(stamps)) > len(stamps) // 2 else rec, np.float64)
    period = float(np.median(np.diff(t))) / 1e9
    rate = round(1.0 / period, 3) if period > 0 else 10.0
    if abs(rate - round(rate)) < 0.02 * rate:    # 20.047 measured = 20 nominal
        rate = float(round(rate))

    profile = {
        "name": name or f"derived from {os.path.basename(bag_path)} {topic}"
                        f"{f' ({frame})' if frame else ''}",
        "cols": cols,
        "layer_elevations_deg": [round(float(v), 4) for v in layers],
        "layer_azimuth0_deg": az0,
        "scan_rate_hz": rate,
    }
    ifield = next((f for f in INTENSITY_FIELDS if f in fields.names), None)
    if ifield:
        kind = fields[ifield]
        if kind.kind == "u" and kind.itemsize <= 2:
            imax = float(np.iinfo(kind).max)
        else:
            seen = max(float(np.nanmax(p[ifield])) for p in scans if len(p))
            imax = next((m for m in (1.0, 255.0, 4095.0, 65535.0) if seen <= m), seen)
        profile["intensity_field"] = ifield
        profile["intensity_max"] = imax
    else:
        profile["intensity_field"] = None

    # How well the recording fits a fixed grid.
    sg_before = sg.PROFILE
    sg.configure(profile)
    share = []
    for p in scans:
        _, _, flat = sg.scan_to_grid(p)
        v = flat[flat >= 0]
        share.append(len(np.unique(v)) / max(len(v), 1))
    if sg_before is not None:
        sg.configure(sg_before)
    elev_spread = float(np.percentile(np.concatenate(spread), 99))
    fit = {
        "elevation_spread_deg_p99": round(elev_spread, 4),
        "min_layer_spacing_deg": round(float(spacing), 4),
        "azimuth_jitter_steps_rms": round(float(np.max(jitter)), 4),
        "points_alone_in_cell": round(float(np.mean(share)), 4),
        "layers_with_other_step": odd,
    }
    problems = []
    if elev_spread > 0.25 * spacing:
        problems.append("points do not sit on distinct elevation layers")
    if max(jitter) > 0.2:
        problems.append("points do not sit on a fixed azimuth step")
    if odd:
        problems.append(f"layers {odd} have another azimuth step "
                        f"({[int(c) for c in layer_cols[odd]]} per turn)")
    say(f"{os.path.basename(bag_path)}  {topic}  frame '{frame}'  "
        f"fields: {', '.join(fields.names)}")
    say(f"  {len(layers)} layers from {layers[0]:.2f} to {layers[-1]:.2f} deg, "
        f"{cols} columns ({step:g} deg), {rate:g} Hz, "
        f"intensity: {ifield or 'none'}")
    say(f"  fit: elevation spread {elev_spread:.3f} deg (layers {spacing:.2f} deg "
        f"apart), azimuth jitter {max(jitter):.3f} step, "
        f"{100 * np.mean(share):.1f}% of points alone in their cell")
    if np.mean(share) < 0.9 and not problems:
        say("  note: several returns share cells; fine for multi-echo recordings "
            "(the nearest is kept), otherwise the grid is too coarse")
    for p in problems:
        say(f"  PROBLEM: {p}")
    profile["_derived"] = {"bag": os.path.basename(bag_path), "topic": topic,
                           "frame": frame, "fit": fit, "problems": problems}
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("-o", "--output", default=None, help="write the profile here")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--name", default=None, help="a label for the sensor")
    ap.add_argument("--scans", type=int, default=40)
    a = ap.parse_args()
    try:
        profile = derive(a.bag, a.topic, a.name, a.scans)
    except ValueError as e:
        sys.exit(str(e))
    text = json.dumps(profile, indent=1)
    if a.output:
        if os.path.exists(a.output):
            sys.exit(f"{a.output} already exists")
        with open(a.output, "w") as f:
            f.write(text + "\n")
        print(f"written to {a.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
