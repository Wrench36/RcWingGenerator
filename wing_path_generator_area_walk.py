#!/usr/bin/env python3
"""
Wing Path Generator - alpha prototype

Generates continuous-path, centerline-based RC wing toolpaths for standing-on-end
3D printed wings. Includes a PyQt6 UI, matplotlib layer viewer, JSON save/load,
and G-code export.

Dependencies:
    pip install PyQt6 matplotlib numpy shapely

Run:
    python wing_path_generator.py

This is an engineering alpha: inspect generated G-code in a previewer before printing.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QDoubleSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView, QSlider,
    QProgressDialog, QSizePolicy
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

try:
    from shapely.geometry import Polygon, Point as ShapelyPoint, MultiPolygon, GeometryCollection
    from shapely.affinity import translate as shapely_translate
except Exception:  # pragma: no cover - handled at runtime
    Polygon = None
    ShapelyPoint = None
    MultiPolygon = None
    GeometryCollection = None
    shapely_translate = None


Point = Tuple[float, float]


@dataclass
class PrinterConfig:
    bed_size_x: float = 220.0
    bed_size_y: float = 220.0
    filament_diameter: float = 1.75
    nozzle_diameter: float = 0.40
    line_width: float = 0.42
    layer_height: float = 0.25
    flow_coeff: float = 1.00
    shell_count: int = 1
    inner_speed: float = 35.0
    outer_speed: float = 28.0
    travel_speed: float = 120.0
    retract_distance: float = 0.8
    retract_speed: float = 35.0
    clearance_travel: float = 5.0
    nozzle_temp: int = 235
    bed_temp: int = 55
    fan_speed: int = 80


@dataclass
class AirfoilConfig:
    source_type: str = "NACA 4-digit"
    source: str = "4412"


@dataclass
class SparConfig:
    enabled: bool = True
    placement_mode: str = "Chord %"
    placement_value: float = 30.0
    diameter: float = 8.0


@dataclass
class RibConfig:
    enabled: bool = True
    spacing: float = 10.0
    angle_deg: float = 45.0
    family_a: bool = True
    family_b: bool = True


@dataclass
class StationConfig:
    z: float = 0.0
    chord: float = 173.0


@dataclass
class GCodeConfig:
    origin_x: float = 0.0
    origin_y: float = 0.0
    z_offset: float = 0.0
    use_relative_e: bool = False
    output_file: str = "wing.gcode"


@dataclass
class AppConfig:
    printer: PrinterConfig = field(default_factory=PrinterConfig)
    airfoil: AirfoilConfig = field(default_factory=AirfoilConfig)
    spars: List[SparConfig] = field(default_factory=lambda: [SparConfig()])
    ribs: RibConfig = field(default_factory=RibConfig)
    stations: List[StationConfig] = field(default_factory=lambda: [StationConfig(0, 173), StationConfig(250, 173), StationConfig(500, 130), StationConfig(750, 75)])
    gcode: GCodeConfig = field(default_factory=GCodeConfig)


@dataclass
class Path:
    name: str
    points: List[Point]
    extrude: bool = True
    speed: float = 30.0


@dataclass
class DebugPrimitive:
    group: str
    geometry: object


@dataclass
class LayerPlan:
    layer_index: int
    z: float
    chord: float
    paths: List[Path]
    valid_ribs: int = 0
    debug_primitives: List[DebugPrimitive] = field(default_factory=list)


class AirfoilProfile:
    def __init__(self, x: np.ndarray, upper: np.ndarray, lower: np.ndarray):
        self.x = x
        self.upper = upper
        self.lower = lower
        self.mid = (upper + lower) / 2.0

    def scaled(self, chord: float) -> "AirfoilProfile":
        return AirfoilProfile(self.x * chord, self.upper * chord, self.lower * chord)

    def y_upper(self, xq: float) -> float:
        return float(np.interp(xq, self.x, self.upper))

    def y_lower(self, xq: float) -> float:
        return float(np.interp(xq, self.x, self.lower))

    def y_mid(self, xq: float) -> float:
        return float(np.interp(xq, self.x, self.mid))


def naca4(code: str, n: int = 500) -> AirfoilProfile:
    code = re.sub(r"\D", "", code)
    if len(code) != 4:
        raise ValueError("NACA 4-digit source must be four digits, e.g. 2412")
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    beta = np.linspace(0, math.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if p > 0 and m > 0:
        left = x < p
        yc[left] = m / p**2 * (2*p*x[left] - x[left]**2)
        dyc[left] = 2*m / p**2 * (p - x[left])
        right = ~left
        yc[right] = m / (1-p)**2 * ((1 - 2*p) + 2*p*x[right] - x[right]**2)
        dyc[right] = 2*m / (1-p)**2 * (p - x[right])
    theta = np.arctan(dyc)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    grid = np.linspace(0, 1, n)
    upper = np.interp(grid, np.sort(xu), yu[np.argsort(xu)])
    lower = np.interp(grid, np.sort(xl), yl[np.argsort(xl)])
    return AirfoilProfile(grid, upper, lower)


def parse_dat_text(text: str, n: int = 500) -> AirfoilProfile:
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
            pts.append((x, y))
        except ValueError:
            continue
    if len(pts) < 10:
        raise ValueError("Could not parse enough airfoil coordinate points")
    arr = np.array(pts, dtype=float)
    # normalize to x 0..1 and chord-length y scale
    xmin, xmax = float(arr[:,0].min()), float(arr[:,0].max())
    chord = xmax - xmin
    if chord <= 0:
        raise ValueError("Airfoil x-coordinates have zero chord")
    arr[:,0] = (arr[:,0] - xmin) / chord
    arr[:,1] = arr[:,1] / chord
    le_idx = int(np.argmin(arr[:,0]))
    s1 = arr[:le_idx+1]
    s2 = arr[le_idx:]
    grid = np.linspace(0, 1, n)

    def interp_surface(seg: np.ndarray) -> np.ndarray:
        # collapse duplicate x values by averaging y
        order = np.argsort(seg[:,0])
        xs = seg[order,0]; ys = seg[order,1]
        ux = [] ; uy = []
        i = 0
        while i < len(xs):
            j = i + 1
            while j < len(xs) and abs(xs[j] - xs[i]) < 1e-9:
                j += 1
            ux.append(float(xs[i:j].mean()))
            uy.append(float(ys[i:j].mean()))
            i = j
        return np.interp(grid, np.array(ux), np.array(uy))

    y1 = interp_surface(s1)
    y2 = interp_surface(s2)
    upper = np.maximum(y1, y2)
    lower = np.minimum(y1, y2)
    return AirfoilProfile(grid, upper, lower)


def load_airfoil(cfg: AirfoilConfig) -> AirfoilProfile:
    if cfg.source_type == "NACA 4-digit":
        return naca4(cfg.source)
    if cfg.source_type == "Local .dat":
        with open(cfg.source, "r", encoding="utf-8", errors="ignore") as f:
            return parse_dat_text(f.read())
    if cfg.source_type == "URL .dat":
        with urllib.request.urlopen(cfg.source, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return parse_dat_text(text)
    raise ValueError(f"Unknown airfoil source type: {cfg.source_type}")


def chord_at(stations: List[StationConfig], z: float) -> float:
    st = sorted(stations, key=lambda s: s.z)
    if not st:
        return 100.0
    if z <= st[0].z:
        return st[0].chord
    if z >= st[-1].z:
        return st[-1].chord
    for a, b in zip(st[:-1], st[1:]):
        if a.z <= z <= b.z:
            if abs(b.z - a.z) < 1e-9:
                return b.chord
            t = (z - a.z) / (b.z - a.z)
            return a.chord * (1-t) + b.chord * t
    return st[-1].chord


def total_span(stations: List[StationConfig]) -> float:
    return max((s.z for s in stations), default=0.0)


def dedupe_points(points: List[Point], eps: float = 1e-6) -> List[Point]:
    out = []
    for p in points:
        if not out or math.hypot(p[0]-out[-1][0], p[1]-out[-1][1]) > eps:
            out.append(p)
    return out


def spar_x_from_config(s: SparConfig, chord: float) -> float:
    if s.placement_mode == "Chord %":
        return chord * s.placement_value / 100.0
    return s.placement_value

def root_chord(cfg: AppConfig) -> float:
    return chord_at(cfg.stations, 0.0)


def spar_global_x_from_config(s: SparConfig, root_chord_value: float) -> float:
    """
    Resolve spar placement into fixed printer-space X, based on the root chord.

    Chord % means percent of the root chord.
    LE distance mm means absolute distance from the root leading edge.
    """
    if s.placement_mode == "Chord %":
        return root_chord_value * s.placement_value / 100.0
    return s.placement_value


def primary_spar_anchor_x(cfg: AppConfig) -> Optional[float]:
    """
    Use the first enabled spar as the taper/scaling anchor.
    If no enabled spar exists, return None and the leading edge remains fixed.
    """
    rc = root_chord(cfg)

    for sp in cfg.spars:
        if sp.enabled and sp.diameter > 0:
            x = spar_global_x_from_config(sp, rc)
            if 0.0 < x < rc:
                return x

    return None


def layer_x_offset(cfg: AppConfig, chord: float) -> float:
    """
    Shift each layer so the airfoil scales around the first enabled spar.

    The anchor keeps the same fraction of the current chord that it had
    at the root, while also staying fixed in printer-space.
    """
    rc = root_chord(cfg)
    if rc <= 0:
        return 0.0

    anchor = primary_spar_anchor_x(cfg)
    if anchor is None:
        return 0.0

    anchor_fraction = anchor / rc
    local_anchor_x = anchor_fraction * chord

    return anchor - local_anchor_x


def translate_points(points: List[Point], dx: float, dy: float = 0.0) -> List[Point]:
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return points
    return [(x + dx, y + dy) for x, y in points]

def model_origin_xy(cfg: AppConfig, airfoil: AirfoilProfile) -> Point:
    """
    Return the model-space origin.

    We define model origin as the root leading-edge center:
        X = root layer leading-edge X after spar-centered taper offset
        Y = root airfoil midline at leading edge

    Exporter Origin X/Y is then applied after subtracting this origin.
    """
    rc = root_chord(cfg)
    root_profile = airfoil.scaled(rc)

    x0 = layer_x_offset(cfg, rc)
    y0 = root_profile.y_mid(0.0)

    return x0, y0


def wing_x_bounds(cfg: AppConfig) -> Tuple[float, float]:
    """
    Approximate global X bounds for preview scaling across all profile stations.
    """
    if not cfg.stations:
        return 0.0, 100.0

    xs_min = []
    xs_max = []

    for st in cfg.stations:
        dx = layer_x_offset(cfg, st.chord)
        xs_min.append(dx)
        xs_max.append(dx + st.chord)

    return min(xs_min), max(xs_max)



def skin_boundary(profile: AirfoilProfile, x: float, top: bool, shell_offset: float) -> float:
    if top:
        return profile.y_upper(x) - shell_offset
    return profile.y_lower(x) + shell_offset


def split_boundary(profile: AirfoilProfile, x: float, top: bool, split_offset: float) -> float:
    return profile.y_mid(x) + (split_offset if top else -split_offset)


def require_shapely():
    if Polygon is None or ShapelyPoint is None:
        raise RuntimeError(
            "This area/walk geometry version requires Shapely. Install it with: pip install shapely"
        )


def clean_geom(g):
    """
    Fix tiny topology artifacts from difference/union operations.
    buffer(0) is the standard Shapely cleanup for near-valid polygons.
    """
    if g is None or g.is_empty:
        return g
    try:
        return g.buffer(0)
    except Exception:
        return g


def largest_polygon(g):
    """
    Return the largest polygonal piece from a Polygon/MultiPolygon/GeometryCollection.
    """
    if g is None or g.is_empty:
        return None

    if Polygon is not None and isinstance(g, Polygon):
        return g

    polys = []

    if MultiPolygon is not None and isinstance(g, MultiPolygon):
        polys.extend(list(g.geoms))
    elif GeometryCollection is not None and isinstance(g, GeometryCollection):
        for part in g.geoms:
            if isinstance(part, Polygon):
                polys.append(part)
            elif MultiPolygon is not None and isinstance(part, MultiPolygon):
                polys.extend(list(part.geoms))

    if not polys:
        return None

    return max(polys, key=lambda poly: poly.area)


def build_half_domain_polygon(
    profile: AirfoilProfile,
    top: bool,
    shell_offset: float,
    split_offset: float,
    chord: float,
    samples: int = 520,
):
    """
    Build the planning area whose boundary is the printable loop.

    This is not a physical wall-thickness solid. It is a 2D planning domain:
    one boundary is the active split spline and the other is the active skin/shell.
    Subtracting primitives from this domain creates boundary notches that are
    then walked as nozzle-centerline paths.
    """
    require_shapely()

    xs_split = np.linspace(chord, 0.0, samples)
    split = [(float(x), split_boundary(profile, float(x), top, split_offset)) for x in xs_split]

    xs_skin = np.linspace(0.0, chord, samples)
    skin = [(float(x), skin_boundary(profile, float(x), top, shell_offset)) for x in xs_skin]

    pts = dedupe_points(split + skin)
    if len(pts) < 4:
        return None

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return largest_polygon(poly)


def generate_rib_starts(chord: float, spacing: float, phase: float) -> List[float]:
    if spacing <= 0:
        return []

    first = -spacing + (phase % spacing)
    xs = []
    x = first

    while x <= chord + spacing:
        if 0.0 < x < chord:
            xs.append(float(x))
        x += spacing

    return xs


def merge_nearby_rib_starts(starts: List[float], line_width: float) -> List[float]:
    if not starts:
        return []

    starts = sorted(starts, reverse=True)
    merged: List[float] = []

    for x in starts:
        if not merged or abs(merged[-1] - x) >= line_width:
            merged.append(x)

    return merged


def rib_phase(ribs: RibConfig, layer_z: float, layer_height: float, line_width: float) -> float:
    ideal = math.tan(math.radians(ribs.angle_deg)) * layer_height
    phase_step = min(abs(ideal), line_width / 2.0)
    layer_index = layer_z / max(layer_height, 1e-9)
    return layer_index * phase_step


def build_rib_slot_primitive(
    profile: AirfoilProfile,
    top: bool,
    x1: float,
    line_width: float,
    chord: float,
    split_offset: float,
    rib_target_skin_offset: float,
    primitive_chord_width: Optional[float] = None,
    primitive_chord_x_shift: float = 0.0,
):
    """
    Make one rectangular/parallelogram-ish slot primitive touching the split
    boundary and intruding toward the active skin/shell boundary.

    Subtracting this primitive from the half-domain makes the exterior boundary
    walk a U-shaped rib intrusion. A rib is suppressed if it cannot satisfy the
    built-in 4-linewidth rule.
    """
    require_shapely()

    if primitive_chord_width is None:
        primitive_chord_width = line_width

    # For two-shell mode the inner rib primitive is intentionally wider and
    # shifted forward/chordwise +X so its reserved/printed area does not sit
    # directly on top of the outer rib primitive.
    x1 = x1 + primitive_chord_x_shift
    x2 = x1 - primitive_chord_width

    if x2 <= 0.0 or x1 >= chord:
        return None

    y_split_1 = split_boundary(profile, x1, top, split_offset)
    y_split_2 = split_boundary(profile, x2, top, split_offset)
    y_target_1 = skin_boundary(profile, x1, top, rib_target_skin_offset + line_width / 2.0)
    y_target_2 = skin_boundary(profile, x2, top, rib_target_skin_offset + line_width / 2.0)

    gap1 = (y_target_1 - y_split_1) if top else (y_split_1 - y_target_1)
    gap2 = (y_target_2 - y_split_2) if top else (y_split_2 - y_target_2)

    if min(gap1, gap2) < 4.0 * line_width:
        return None

    pts = [
        (x1, y_split_1),
        (x1, y_target_1),
        (x2, y_target_2),
        (x2, y_split_2),
    ]

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or poly.area <= 0.0:
        return None

    return poly


def build_rib_primitives(
    profile: AirfoilProfile,
    top: bool,
    ribs: RibConfig,
    line_width: float,
    layer_z: float,
    layer_height: float,
    chord: float,
    split_offset: float,
    rib_target_skin_offset: float,
    primitive_chord_width: Optional[float] = None,
    primitive_chord_x_shift: float = 0.0,
) -> Tuple[List[object], int]:
    if not ribs.enabled or ribs.spacing <= 0.0:
        return [], 0

    phase = rib_phase(ribs, layer_z, layer_height, line_width)
    starts: List[float] = []

    if ribs.family_a:
        starts.extend(generate_rib_starts(chord, ribs.spacing, phase))

    if ribs.family_b:
        starts.extend(generate_rib_starts(chord, ribs.spacing, -phase))

    starts = merge_nearby_rib_starts(starts, line_width)
    prims = []

    for x in starts:
        prim = build_rib_slot_primitive(
            profile=profile,
            top=top,
            x1=x,
            line_width=line_width,
            chord=chord,
            split_offset=split_offset,
            rib_target_skin_offset=rib_target_skin_offset,
            primitive_chord_width=primitive_chord_width,
            primitive_chord_x_shift=primitive_chord_x_shift,
        )

        if prim is not None:
            prims.append(prim)

    return prims, len(prims)


def spar_primitive_fits(
    profile: AirfoilProfile,
    top: bool,
    cx: float,
    r: float,
    active_skin_offset: float,
    line_width: float,
    chord: float,
    samples: int = 36,
) -> bool:
    """
    Apply the built-in 4-linewidth rule to spar pockets.

    The user is responsible for placing spars that physically fit. This only
    suppresses a pocket when it would create tiny unprintable fragments near the
    active skin/shell boundary.
    """
    if cx - r <= 0.0 or cx + r >= chord:
        return False

    if top:
        angles = np.linspace(0.0, math.pi, samples)
    else:
        angles = np.linspace(0.0, -math.pi, samples)

    cy = profile.y_mid(cx)
    min_gap = float("inf")

    for a in angles:
        x = cx + r * math.cos(float(a))
        y = cy + r * math.sin(float(a))

        if x <= 0.0 or x >= chord:
            return False

        boundary_y = skin_boundary(profile, x, top, active_skin_offset + line_width / 2.0)
        gap = (boundary_y - y) if top else (y - boundary_y)
        min_gap = min(min_gap, gap)

    return min_gap >= 4.0 * line_width


def build_spar_primitives(
    profile: AirfoilProfile,
    top: bool,
    cfg: AppConfig,
    chord: float,
    x_offset: float,
    active_skin_offset: float,
    radius_extra: float = 0.0,
) -> List[object]:
    require_shapely()

    prims = []
    rc = root_chord(cfg)
    lw = cfg.printer.line_width

    for sp in cfg.spars:
        if not sp.enabled or sp.diameter <= 0.0:
            continue

        global_cx = spar_global_x_from_config(sp, rc)
        cx = global_cx - x_offset
        r = sp.diameter / 2.0 + radius_extra

        if not spar_primitive_fits(profile, top, cx, r, active_skin_offset, lw, chord):
            continue

        cy = profile.y_mid(cx)
        prims.append(ShapelyPoint(cx, cy).buffer(r, resolution=32))


        if radius_extra > 0.0:
            # Inner shell
            rib_target_skin_offset = lw
            primitive_chord_width = lw * 3.0
            primitive_chord_x_shift = lw
        else:
            # Outer shell
            primitive_chord_width = lw
            primitive_chord_x_shift = 0.0

            if cfg.printer.shell_count >= 2:
                rib_target_skin_offset = lw * 2.0
            else:
                rib_target_skin_offset = 0.0

        webs = build_spar_web_primitives(
            profile=profile,
            top=top,
            cfg=cfg,
            chord=chord,
            x_offset=x_offset,
            active_skin_offset=active_skin_offset,
            split_offset=lw / 2.0,
            rib_target_skin_offset=rib_target_skin_offset,
            primitive_chord_width=primitive_chord_width,
            primitive_chord_x_shift=primitive_chord_x_shift,
            radius_extra=radius_extra,
        )
        for web in webs:
            prims.append(web)

    return prims

def build_spar_web_primitives(
    profile: AirfoilProfile,
    top: bool,
    cfg: AppConfig,
    chord: float,
    x_offset: float,
    active_skin_offset: float,
    split_offset: float,
    rib_target_skin_offset: float,
    primitive_chord_width: float,
    primitive_chord_x_shift: float,
    radius_extra: float = 0.0,
) -> List[object]:
    require_shapely()

    prims = []
    rc = root_chord(cfg)
    line_width = cfg.printer.line_width

    for sp in cfg.spars:
        if not sp.enabled or sp.diameter <= 0.0:
            continue

        global_cx = spar_global_x_from_config(sp, rc)
        cx = global_cx - x_offset
        r = sp.diameter / 2.0 + radius_extra

        if not spar_primitive_fits(profile, top, cx, r, active_skin_offset, line_width, chord):
            continue

        # First try the spar centerline. If the web visually intersects weirdly,
        # change this to cx + r or cx - r after previewing.
        x = cx

        prim = build_rib_slot_primitive(
            profile=profile,
            top=top,
            x1=x,
            line_width=line_width,
            chord=chord,
            split_offset=split_offset,
            rib_target_skin_offset=rib_target_skin_offset,
            primitive_chord_width=primitive_chord_width,
            primitive_chord_x_shift=primitive_chord_x_shift,
        )

        if prim is not None and not prim.is_empty:
            prims.append(prim)

    return prims

def subtract_primitives(domain, primitives: List[object]):
    g = domain

    for prim in primitives:
        if prim is None or prim.is_empty:
            continue

        g = g.difference(prim)
        g = clean_geom(g)

        if g is None or g.is_empty:
            return None

    return largest_polygon(g)


def ring_signed_area(points: List[Point]) -> float:
    if len(points) < 3:
        return 0.0

    area = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        area += a[0] * b[1] - b[0] * a[1]

    return area / 2.0


def rotate_ring_to_index(points: List[Point], idx: int) -> List[Point]:
    if not points:
        return []
    return points[idx:] + points[:idx]


def orient_half_loop(points: List[Point], top: bool, profile: AirfoilProfile, split_offset: float, chord: float) -> List[Point]:
    """
    Polygon boundaries can start anywhere and may be CW/CCW.
    Rotate/reverse so the path begins near the trailing-edge split point and
    proceeds along the split/internal/notched side first, skin side last.
    """
    pts = dedupe_points(points)

    if len(pts) > 2 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-7:
        pts = pts[:-1]

    if len(pts) < 2:
        return pts

    start = (chord, split_boundary(profile, chord, top, split_offset))

    def d2(p: Point) -> float:
        return (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2

    idx = min(range(len(pts)), key=lambda i: d2(pts[i]))
    forward = rotate_ring_to_index(pts, idx)
    reverse_src = list(reversed(pts))
    ridx = min(range(len(reverse_src)), key=lambda i: d2(reverse_src[i]))
    reverse = rotate_ring_to_index(reverse_src, ridx)

    def score(candidate: List[Point]) -> float:
        if len(candidate) < 4:
            return float("inf")

        # Prefer first few moves to go LE-ward along split/internal side.
        n = min(12, len(candidate) - 1)
        dx_sum = sum(candidate[i + 1][0] - candidate[i][0] for i in range(n))

        # Prefer first segment y to be nearer split than skin at that x.
        sample = candidate[min(4, len(candidate) - 1)]
        x = max(0.0, min(chord, sample[0]))
        split_y = split_boundary(profile, x, top, split_offset)
        skin_y = skin_boundary(profile, x, top, 0.0)
        ds = abs(sample[1] - split_y)
        dk = abs(sample[1] - skin_y)

        return dx_sum + (0.25 if ds > dk else 0.0)

    chosen = forward if score(forward) <= score(reverse) else reverse
    return dedupe_points(chosen + [chosen[0]])


def walk_region_boundary(region, top: bool, profile: AirfoilProfile, split_offset: float, chord: float) -> List[Point]:
    poly = largest_polygon(region)

    if poly is None or poly.is_empty:
        return []

    coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
    return orient_half_loop(coords, top, profile, split_offset, chord)


def make_half_area_loop(
    profile: AirfoilProfile,
    top: bool,
    shell_offset: float,
    split_offset: float,
    cfg: AppConfig,
    z: float,
    chord: float,
    include_ribs: bool,
    x_offset: float,
    rib_target_skin_offset: Optional[float] = None,
    rib_primitive_chord_width: Optional[float] = None,
    rib_primitive_chord_x_shift: float = 0.0,
    spar_radius_extra: float = 0.0,
) -> Tuple[List[Point], int]:
    """
    Area/primitive/walk path generator.

    1. Build the half-airfoil planning area between active split and active skin.
    2. Add spar pockets and rib slots as primitives.
    3. Subtract those primitives from the area.
    4. Walk the resulting exterior boundary as the nozzle-centerline path.
    """
    require_shapely()

    lw = cfg.printer.line_width
    if rib_target_skin_offset is None:
        rib_target_skin_offset = shell_offset

    domain = build_half_domain_polygon(
        profile=profile,
        top=top,
        shell_offset=shell_offset,
        split_offset=split_offset,
        chord=chord,
    )

    if domain is None or domain.is_empty:
        return [], 0

    primitives: List[object] = []
    primitives.extend(build_spar_primitives(
        profile,
        top,
        cfg,
        chord,
        x_offset,
        active_skin_offset=shell_offset,
        radius_extra=spar_radius_extra,
    ))

    valid_ribs = 0
    if include_ribs:
        rib_prims, valid_ribs = build_rib_primitives(
            profile=profile,
            top=top,
            ribs=cfg.ribs,
            line_width=lw,
            layer_z=z,
            layer_height=cfg.printer.layer_height,
            chord=chord,
            split_offset=split_offset,
            rib_target_skin_offset=rib_target_skin_offset,
            primitive_chord_width=rib_primitive_chord_width,
            primitive_chord_x_shift=rib_primitive_chord_x_shift,
        )
        primitives.extend(rib_prims)

    region = subtract_primitives(domain, primitives)

    if region is None or region.is_empty:
        return [], 0

    pts = walk_region_boundary(region, top, profile, split_offset, chord)
    return translate_points(dedupe_points(pts), x_offset), valid_ribs



def clip_and_translate_primitives(primitives: List[object], domain, x_offset: float) -> List[object]:
    """
    Return the actual primitive area that participates in the current shell/half.

    This is the debug shape we want to see: the primitive clipped to the active
    half-domain, then shifted into model/global X by the same x_offset as the
    final walked path.
    """
    out = []

    if shapely_translate is None or domain is None or domain.is_empty:
        return out

    for prim in primitives:
        if prim is None or prim.is_empty:
            continue

        g = prim.intersection(domain)
        g = clean_geom(g)

        if g is None or g.is_empty:
            continue

        out.append(shapely_translate(g, xoff=x_offset, yoff=0.0))

    return out


def collect_half_debug_primitives(
    profile: AirfoilProfile,
    top: bool,
    shell_group: str,
    shell_offset: float,
    split_offset: float,
    cfg: AppConfig,
    z: float,
    chord: float,
    x_offset: float,
    rib_target_skin_offset: float,
    rib_primitive_chord_width: Optional[float] = None,
    rib_primitive_chord_x_shift: float = 0.0,
    spar_radius_extra: float = 0.0,
) -> Tuple[List[DebugPrimitive], int]:
    """
    Build debug-only primitive geometry for one half/shell.

    The returned geometries are clipped to the same active half-domain used for
    path walking. Ribs are therefore the final printable/clipping result, not
    raw candidates. Spars are the spar-pocket areas that actually intersect the
    current shell/half.
    """
    require_shapely()

    lw = cfg.printer.line_width
    debug: List[DebugPrimitive] = []
    valid_ribs = 0

    domain = build_half_domain_polygon(
        profile=profile,
        top=top,
        shell_offset=shell_offset,
        split_offset=split_offset,
        chord=chord,
    )

    if domain is None or domain.is_empty:
        return debug, 0

    spar_prims = build_spar_primitives(
        profile=profile,
        top=top,
        cfg=cfg,
        chord=chord,
        x_offset=x_offset,
        active_skin_offset=shell_offset,
        radius_extra=spar_radius_extra,
    )

    for g in clip_and_translate_primitives(spar_prims, domain, x_offset):
        debug.append(DebugPrimitive(f"{shell_group}_spar_pockets", g))

    if cfg.ribs.enabled:
        rib_prims, valid_ribs = build_rib_primitives(
            profile=profile,
            top=top,
            ribs=cfg.ribs,
            line_width=lw,
            layer_z=z,
            layer_height=cfg.printer.layer_height,
            chord=chord,
            split_offset=split_offset,
            rib_target_skin_offset=rib_target_skin_offset,
            primitive_chord_width=rib_primitive_chord_width,
            primitive_chord_x_shift=rib_primitive_chord_x_shift,
        )

        for g in clip_and_translate_primitives(rib_prims, domain, x_offset):
            debug.append(DebugPrimitive(f"{shell_group}_ribs", g))

    return debug, valid_ribs


def build_layer_debug_primitives(
    base_airfoil: AirfoilProfile,
    cfg: AppConfig,
    layer_index: int,
) -> List[DebugPrimitive]:
    """
    Build semi-transparent primitive groups for the matplotlib viewer.

    These are not consumed by the G-code writer. They are for debugging the
    area/primitive/walk geometry only.
    """
    require_shapely()

    z = layer_index * cfg.printer.layer_height
    span = total_span(cfg.stations)

    if span > 0:
        z = min(z, span)

    chord = chord_at(cfg.stations, z)
    profile = base_airfoil.scaled(chord)
    lw = cfg.printer.line_width
    x_offset = layer_x_offset(cfg, chord)

    debug: List[DebugPrimitive] = []

    if cfg.printer.shell_count >= 2:
        inner_split_offset = lw * 1.5
        inner_skin_offset = lw
        outer_split_offset = lw / 2.0
        outer_skin_offset = 0.0
        outer_rib_target = lw*2.0

        for top in (True, False):
            d, _ = collect_half_debug_primitives(
                profile=profile,
                top=top,
                shell_group="inner",
                shell_offset=inner_skin_offset,
                split_offset=inner_split_offset,
                cfg=cfg,
                z=z,
                chord=chord,
                x_offset=x_offset,
                rib_target_skin_offset=inner_skin_offset,
                rib_primitive_chord_width=lw * 3.0,
                rib_primitive_chord_x_shift=lw,
                spar_radius_extra=lw,
            )
            debug.extend(d)

            d, _ = collect_half_debug_primitives(
                profile=profile,
                top=top,
                shell_group="outer",
                shell_offset=outer_skin_offset,
                split_offset=outer_split_offset,
                cfg=cfg,
                z=z,
                chord=chord,
                x_offset=x_offset,
                rib_target_skin_offset=outer_rib_target,
            )
            debug.extend(d)
    else:
        split_offset = lw / 2.0

        for top in (True, False):
            d, _ = collect_half_debug_primitives(
                profile=profile,
                top=top,
                shell_group="outer",
                shell_offset=0.0,
                split_offset=split_offset,
                cfg=cfg,
                z=z,
                chord=chord,
                x_offset=x_offset,
                rib_target_skin_offset=0.0,
            )
            debug.extend(d)

    return debug


def generate_layer(base_airfoil: AirfoilProfile, cfg: AppConfig, layer_index: int) -> LayerPlan:
    z = layer_index * cfg.printer.layer_height
    span = total_span(cfg.stations)

    if span > 0:
        z = min(z, span)

    chord = chord_at(cfg.stations, z)
    profile = base_airfoil.scaled(chord)

    lw = cfg.printer.line_width
    x_offset = layer_x_offset(cfg, chord)

    paths: List[Path] = []
    valid_ribs = 0

    if cfg.printer.shell_count >= 2:
        # Inner shell: offset from both the skin and split boundaries.
        inner_split_offset = lw * 1.5
        inner_skin_offset = lw

        top_inner, n1 = make_half_area_loop(
            profile=profile,
            top=True,
            shell_offset=inner_skin_offset,
            split_offset=inner_split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=inner_skin_offset,
            rib_primitive_chord_width=lw * 3.0,
            rib_primitive_chord_x_shift=lw,
            spar_radius_extra=lw,
        )

        bottom_inner, n2 = make_half_area_loop(
            profile=profile,
            top=False,
            shell_offset=inner_skin_offset,
            split_offset=inner_split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=inner_skin_offset,
            rib_primitive_chord_width=lw * 3.0,
            rib_primitive_chord_x_shift=lw,
            spar_radius_extra=lw,
        )

        valid_ribs += n1 + n2

        # Outer shell: true outer skin, but ribs are shortened to stop before
        # the inner-shell space. This keeps the outer visible skin pass clean
        # while preserving the same top/bottom path family split.
        outer_split_offset = lw / 2.0
        outer_skin_offset = 0.0
        outer_rib_target = lw*2.0

        top_outer, n3 = make_half_area_loop(
            profile=profile,
            top=True,
            shell_offset=outer_skin_offset,
            split_offset=outer_split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=outer_rib_target,
        )

        bottom_outer, n4 = make_half_area_loop(
            profile=profile,
            top=False,
            shell_offset=outer_skin_offset,
            split_offset=outer_split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=outer_rib_target,
        )

        valid_ribs += n3 + n4

        if top_inner:
            paths.append(Path("top_inner", top_inner, True, cfg.printer.inner_speed))
        if top_inner and bottom_inner:
            paths.append(Path("travel", clearance_move(top_inner[-1], bottom_inner[0], cfg.printer.clearance_travel), False, cfg.printer.travel_speed))
        if bottom_inner:
            paths.append(Path("bottom_inner", bottom_inner, True, cfg.printer.inner_speed))
        if bottom_inner and top_outer:
            paths.append(Path("travel", clearance_move(bottom_inner[-1], top_outer[0], cfg.printer.clearance_travel), False, cfg.printer.travel_speed))
        if top_outer:
            paths.append(Path("top_outer", top_outer, True, cfg.printer.outer_speed))
        if top_outer and bottom_outer:
            paths.append(Path("travel", clearance_move(top_outer[-1], bottom_outer[0], cfg.printer.clearance_travel), False, cfg.printer.travel_speed))
        if bottom_outer:
            paths.append(Path("bottom_outer", bottom_outer, True, cfg.printer.outer_speed))

    else:
        split_offset = lw / 2.0

        top_outer, n1 = make_half_area_loop(
            profile=profile,
            top=True,
            shell_offset=0.0,
            split_offset=split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=0.0,
        )

        bottom_outer, n2 = make_half_area_loop(
            profile=profile,
            top=False,
            shell_offset=0.0,
            split_offset=split_offset,
            cfg=cfg,
            z=z,
            chord=chord,
            include_ribs=True,
            x_offset=x_offset,
            rib_target_skin_offset=0.0,
        )

        valid_ribs += n1 + n2

        if top_outer:
            paths.append(Path("top_outer", top_outer, True, cfg.printer.outer_speed))
        if top_outer and bottom_outer:
            paths.append(Path("travel", clearance_move(top_outer[-1], bottom_outer[0], cfg.printer.clearance_travel), False, cfg.printer.travel_speed))
        if bottom_outer:
            paths.append(Path("bottom_outer", bottom_outer, True, cfg.printer.outer_speed))

    debug_primitives = build_layer_debug_primitives(base_airfoil, cfg, layer_index)
    return LayerPlan(layer_index, z, chord, paths, valid_ribs, debug_primitives)


def clearance_move(a: Point, b: Point, clearance: float) -> List[Point]:
    # Move aft of trailing edge by clearance, then to next start.
    x = max(a[0], b[0]) + clearance
    return [a, (x, a[1]), (x, b[1]), b]


def path_length(points: List[Point]) -> float:
    return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points[:-1], points[1:]))

def point_line_distance(p: Point, a: Point, b: Point) -> float:
    """
    Perpendicular distance from point p to line segment a-b.
    """
    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    cx = ax + t * dx
    cy = ay + t * dy

    return math.hypot(px - cx, py - cy)


def simplify_polyline_rdp(points: List[Point], tolerance: float) -> List[Point]:
    """
    Simplify a polyline using Ramer-Douglas-Peucker.

    Keeps first/last points and removes intermediate points whose deviation
    from the simplified path is below tolerance.
    """
    if tolerance <= 0.0 or len(points) <= 2:
        return points

    max_dist = -1.0
    max_idx = -1

    a = points[0]
    b = points[-1]

    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], a, b)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > tolerance:
        left = simplify_polyline_rdp(points[:max_idx + 1], tolerance)
        right = simplify_polyline_rdp(points[max_idx:], tolerance)
        return left[:-1] + right

    return [points[0], points[-1]]


def simplify_path_for_gcode(points: List[Point], line_width: float, extrude: bool) -> List[Point]:
    """
    Reduce G-code size while preserving geometry.

    Travel paths are left alone because their intermediate clearance points
    are intentional.

    Extrusion paths get a tolerance based on line width. This keeps airfoil
    curves smooth enough for printing while removing arbitrarily tiny moves.
    """
    if len(points) <= 4:
        return points

    if not extrude:
        return points

    tolerance = max(0.025, min(0.12, line_width * 0.15))

    return dedupe_points(simplify_polyline_rdp(points, tolerance))

def all_model_bounds(cfg: AppConfig, airfoil: AirfoilProfile) -> Tuple[float, float, float, float]:
    """
    Approximate model-space XY bounds by sampling several layers.
    Used only for export comments and sanity checking.
    """
    p = cfg.printer
    span = total_span(cfg.stations)
    layers = max(1, int(math.ceil(span / max(p.layer_height, 1e-9))))

    sample_layers = sorted(set([
        0,
        layers,
        layers // 4,
        layers // 2,
        (3 * layers) // 4,
    ]))

    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")

    for li in sample_layers:
        plan = generate_layer(airfoil, cfg, li)
        for path in plan.paths:
            for x, y in path.points:
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)

    if not math.isfinite(xmin):
        return 0.0, 0.0, 0.0, 0.0

    return xmin, xmax, ymin, ymax

class GCodeWriter:
    def __init__(self, cfg: AppConfig, airfoil: AirfoilProfile, progress_callback=None, mirror_x: bool = False):
        self.cfg = cfg
        self.airfoil = airfoil
        self.progress_callback = progress_callback
        self.mirror_x = mirror_x
        self.model_origin = model_origin_xy(cfg, airfoil)
        self.e = 0.0
        self.lines: List[str] = []
        bx0, bx1, by0, by1 = all_model_bounds(cfg, airfoil)
        ox, oy = self.model_origin

        self.local_xmin = bx0 - ox
        self.local_xmax = bx1 - ox
        self.local_ymin = by0 - oy
        self.local_ymax = by1 - oy

    def extrusion_for_length(self, length: float) -> float:
        p = self.cfg.printer
        vol = length * p.line_width * p.layer_height * p.flow_coeff
        filament_area = math.pi * (p.filament_diameter / 2.0) ** 2
        return vol / filament_area

    def xy(self, p: Point) -> Tuple[float, float]:
        ox, oy = self.model_origin

        local_x = p[0] - ox
        local_y = p[1] - oy

        if self.mirror_x:
            local_x = self.local_xmin + self.local_xmax - local_x

        return (
            local_x + self.cfg.gcode.origin_x,
            local_y + self.cfg.gcode.origin_y,
        )

    def add(self, line: str):
        self.lines.append(line)

    def write(self) -> str:
        p = self.cfg.printer
        span = total_span(self.cfg.stations)
        layers = max(1, int(math.ceil(span / max(p.layer_height, 1e-9))))
        self.add("; Wing Path Generator alpha")
        self.add("; Inspect before printing")
        self.add(f"; X mirror {'ON' if self.mirror_x else 'OFF'}")
        bx0, bx1, by0, by1 = all_model_bounds(self.cfg, self.airfoil)

        self.add(f"; Model origin root LE center X{self.model_origin[0]:.3f} Y{self.model_origin[1]:.3f}")
        self.add(f"; Export origin target X{self.cfg.gcode.origin_x:.3f} Y{self.cfg.gcode.origin_y:.3f}")
        #self.add(f"; Export transform dX{self.export_dx:.3f} dY{self.export_dy:.3f}")
        self.add(
            f"; Approx model bounds X{bx0:.3f}..{bx1:.3f} "
            f"Y{by0:.3f}..{by1:.3f}"
        )
        self.add(
            f"; Approx exported bounds "
            #f"X{bx0 + self.export_dx:.3f}..{bx1 + self.export_dx:.3f} "
            #f"Y{by0 + self.export_dy:.3f}..{by1 + self.export_dy:.3f}"
        )
        self.add("G28 ; home")
        self.add("G21 ; mm")
        self.add("G90 ; absolute XYZ")
        self.add("M82 ; absolute extrusion" if not self.cfg.gcode.use_relative_e else "M83 ; relative extrusion")
        self.add(f"M104 S{p.nozzle_temp}")
        self.add(f"M140 S{p.bed_temp}")
        self.add(f"M109 S{p.nozzle_temp}")
        self.add(f"M190 S{p.bed_temp}")
        self.add(f"M106 S{max(0, min(255, int(p.fan_speed / 100 * 255)))}")
        self.add("G92 E0")
        self.add("G0 Z1")
        for li in range(layers + 1):
            if self.progress_callback is not None:
                if not self.progress_callback(li, layers):
                    raise RuntimeError("G-code export canceled")
            plan = generate_layer(self.airfoil, self.cfg, li)
            zprint = plan.z + self.cfg.gcode.z_offset
            self.add(f";LAYER {li} span_z={plan.z:.3f} chord={plan.chord:.3f}")
            self.add(f"G0 Z{zprint:.3f} F{p.travel_speed*60:.0f}")
            for path in plan.paths:
                points = simplify_path_for_gcode(
                    path.points,
                    self.cfg.printer.line_width,
                    path.extrude,
                )

                if len(points) < 2:
                    continue

                start = points[0]
                x, y = self.xy(start)
                if path.extrude:
                    if p.retract_distance > 0:
                        r = p.retract_distance
                        if self.cfg.gcode.use_relative_e:
                            self.add(f"G1 E{-r:.5f} F{p.retract_speed*60:.0f}")
                        else:
                            self.e -= r
                            self.add(f"G1 E{self.e:.5f} F{p.retract_speed*60:.0f}")
                    self.add(f"G0 X{x:.3f} Y{y:.3f} F{p.travel_speed*60:.0f}")
                    if p.retract_distance > 0:
                        r = p.retract_distance
                        if self.cfg.gcode.use_relative_e:
                            self.add(f"G1 E{r:.5f} F{p.retract_speed*60:.0f}")
                        else:
                            self.e += r
                            self.add(f"G1 E{self.e:.5f} F{p.retract_speed*60:.0f}")
                    for a, b in zip(points[:-1], points[1:]):
                        seg = math.hypot(b[0]-a[0], b[1]-a[1])
                        de = self.extrusion_for_length(seg)
                        bx, by = self.xy(b)
                        if self.cfg.gcode.use_relative_e:
                            self.add(f"G1 X{bx:.3f} Y{by:.3f} E{de:.5f} F{path.speed*60:.0f}")
                        else:
                            self.e += de
                            self.add(f"G1 X{bx:.3f} Y{by:.3f} E{self.e:.5f} F{path.speed*60:.0f}")
                else:
                    self.add(f"G0 X{x:.3f} Y{y:.3f} F{p.travel_speed*60:.0f}")
                    for b in points[1:]:
                        bx, by = self.xy(b)
                        self.add(f"G0 X{bx:.3f} Y{by:.3f} F{path.speed*60:.0f}")
        self.add("M104 S0")
        self.add("M140 S0")
        self.add("M106 S0")
        self.add("G91")
        self.add("G0 Z5 F1800")
        self.add("G90")
        self.add("M84")
        return "\n".join(self.lines) + "\n"


class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self.info = QLabel("No airfoil loaded")
        self.layer_spin = QSpinBox()
        self.layer_spin.setMinimum(0)
        self.layer_spin.setMaximum(999999)
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(999999)
        self.layer_slider.setSingleStep(1)
        self.layer_slider.setPageStep(10)
        self.show_top_outer = QCheckBox("Top outer")
        self.show_top_inner = QCheckBox("Top inner")
        self.show_bottom_outer = QCheckBox("Bottom outer")
        self.show_bottom_inner = QCheckBox("Bottom inner")
        self.show_travel = QCheckBox("Travel")
        self.show_inner_ribs = QCheckBox("Inner ribs")
        self.show_outer_ribs = QCheckBox("Outer ribs")
        self.show_inner_spars = QCheckBox("Inner spar pockets")
        self.show_outer_spars = QCheckBox("Outer spar pockets")
        for cb in [
            self.show_top_outer, self.show_top_inner, self.show_bottom_outer,
            self.show_bottom_inner, self.show_travel, #self.show_inner_ribs,
            #self.show_outer_ribs, self.show_inner_spars, self.show_outer_spars,
        ]:
            cb.setChecked(True)
        top = QHBoxLayout()
        #top.addWidget(QLabel("Layer")); top.addWidget(self.layer_spin); top.addStretch(1)
        top.addWidget(QLabel("Layer"))
        top.addWidget(self.layer_spin)
        top.addWidget(self.layer_slider, 1)
        top.addStretch(1)
        for cb in [self.show_top_outer, self.show_top_inner, self.show_bottom_outer, self.show_bottom_inner, self.show_travel]:
            top.addWidget(cb)

        debug_row = QHBoxLayout()
        debug_row.addWidget(QLabel("Primitives"))
        debug_row.addWidget(self.show_inner_ribs)
        debug_row.addWidget(self.show_outer_ribs)
        debug_row.addWidget(self.show_inner_spars)
        debug_row.addWidget(self.show_outer_spars)
        debug_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(debug_row)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.info)
        self._cfg: Optional[AppConfig] = None
        self._airfoil: Optional[AirfoilProfile] = None
        #self.layer_spin.valueChanged.connect(self.redraw)
        #for cb in [self.show_top_outer, self.show_top_inner, self.show_bottom_outer, self.show_bottom_inner, self.show_travel]:
            #cb.stateChanged.connect(self.redraw)
        self.layer_spin.valueChanged.connect(self.layer_slider.setValue)
        self.layer_slider.valueChanged.connect(self.layer_spin.setValue)
        self.layer_spin.valueChanged.connect(self.redraw)

        for cb in [
            self.show_top_outer, self.show_top_inner, self.show_bottom_outer,
            self.show_bottom_inner, self.show_travel, self.show_inner_ribs,
            self.show_outer_ribs, self.show_inner_spars, self.show_outer_spars,
        ]:
            cb.stateChanged.connect(self.redraw)

    def set_model(self, cfg: AppConfig, airfoil: AirfoilProfile):
        self._cfg = cfg
        self._airfoil = airfoil
        max_layer = int(math.ceil(total_span(cfg.stations) / max(cfg.printer.layer_height, 1e-9)))
        #self.layer_spin.setMaximum(max(0, max_layer))
        #self.redraw()
        max_layer = max(0, max_layer)
        self.layer_spin.setMaximum(max_layer)
        self.layer_slider.setMaximum(max_layer)

        if self.layer_spin.value() > max_layer:
            self.layer_spin.setValue(max_layer)

        self.redraw()

    def visible(self, name: str) -> bool:
        if name == "top_outer": return self.show_top_outer.isChecked()
        if name == "top_inner": return self.show_top_inner.isChecked()
        if name == "bottom_outer": return self.show_bottom_outer.isChecked()
        if name == "bottom_inner": return self.show_bottom_inner.isChecked()
        if name == "travel": return self.show_travel.isChecked()
        return True

    def primitive_visible(self, group: str) -> bool:
        if group == "inner_ribs": return self.show_inner_ribs.isChecked()
        if group == "outer_ribs": return self.show_outer_ribs.isChecked()
        if group == "inner_spar_pockets": return self.show_inner_spars.isChecked()
        if group == "outer_spar_pockets": return self.show_outer_spars.isChecked()
        return False

    def iter_polygons(self, geom):
        if geom is None or geom.is_empty:
            return

        if Polygon is not None and isinstance(geom, Polygon):
            yield geom
            return

        if MultiPolygon is not None and isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                yield poly
            return

        if GeometryCollection is not None and isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                yield from self.iter_polygons(part)

    def draw_debug_primitives(self, plan: LayerPlan):
        style = {
            "inner_ribs": dict(facecolor="tab:cyan", edgecolor="tab:cyan", alpha=0.22, linewidth=0.7),
            "outer_ribs": dict(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.18, linewidth=0.7),
            "inner_spar_pockets": dict(facecolor="tab:red", edgecolor="tab:red", alpha=0.18, linewidth=0.8),
            "outer_spar_pockets": dict(facecolor="tab:purple", edgecolor="tab:purple", alpha=0.16, linewidth=0.8),
        }

        labeled = set()

        for prim in plan.debug_primitives:
            if not self.primitive_visible(prim.group):
                continue

            kwargs = dict(style.get(prim.group, {}))
            label = prim.group.replace("_", " ")

            if label in labeled:
                label = None
            else:
                labeled.add(label)

            for poly in self.iter_polygons(prim.geometry):
                exterior = np.array(poly.exterior.coords)
                if len(exterior) < 3:
                    continue
                patch = MplPolygon(exterior, closed=True, label=label, **kwargs)
                self.ax.add_patch(patch)
                label = None

    def redraw(self):
        self.ax.clear()
        if self._cfg is None or self._airfoil is None:
            self.canvas.draw_idle()
            return
        cfg = self._cfg
        layer = self.layer_spin.value()
        plan = generate_layer(self._airfoil, cfg, layer)
        style = {
            "top_outer": dict(color="tab:blue", linewidth=1.4),
            "top_inner": dict(color="tab:cyan", linewidth=1.2),
            "bottom_outer": dict(color="tab:orange", linewidth=1.4),
            "bottom_inner": dict(color="tab:red", linewidth=1.2),
            "travel": dict(color="gray", linewidth=0.8, linestyle="--"),
        }
        self.draw_debug_primitives(plan)

        for path in plan.paths:
            if not self.visible(path.name):
                continue
            pts = np.array(path.points)
            if len(pts) == 0:
                continue
            self.ax.plot(pts[:,0], pts[:,1], label=path.name, **style.get(path.name, {}))
            if len(pts) > 2 and path.extrude:
                self.ax.scatter([pts[0,0]], [pts[0,1]], s=16)
        root_chord = chord_at(cfg.stations, 0)
        root_profile = self._airfoil.scaled(root_chord)
        ymin = min(float(root_profile.lower.min()) - 8, -10)
        ymax = max(float(root_profile.upper.max()) + 8, 10)
        #self.ax.set_xlim(-5, root_chord + cfg.printer.clearance_travel + 10)
        xmin, xmax = wing_x_bounds(cfg)
        self.ax.set_xlim(xmin - 5, xmax + cfg.printer.clearance_travel + 10)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("Chord X / mm")
        self.ax.set_ylabel("Airfoil Y / mm")
        self.ax.legend(loc="upper right", fontsize=8)
        self.info.setText(f"Layer {plan.layer_index} | span/Z {plan.z:.3f} mm | chord {plan.chord:.3f} mm | valid rib intrusions {plan.valid_ribs}")
        self.canvas.draw_idle()

class HeightResizableTableWidget(QTableWidget):
    """
    QTableWidget with a draggable bottom resize band.

    Drag near the bottom edge of the table to resize its height.
    Handles both table-frame events and viewport events, because QTableWidget
    sends most mouse interaction to its internal viewport.
    """
    def __init__(self, *args, min_h: int = 110, max_h: int = 500, **kwargs):
        super().__init__(*args, **kwargs)

        self._resizing_height = False
        self._resize_start_y = 0
        self._resize_start_h = 0
        self._resize_band_px = 10
        self._min_h = min_h
        self._max_h = max_h

        self.setMinimumHeight(min_h)
        self.setMaximumHeight(max_h)
        self.setFixedHeight(min_h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

    """def _near_bottom_edge_from_global_y(self, global_y: int) -> bool:
        local = self.mapFromGlobal(self.mapToGlobal(self.rect().topLeft()))
        return False"""

    def _near_bottom_edge(self, global_pos) -> bool:
        p = self.mapFromGlobal(global_pos.toPoint())
        return self.height() - self._resize_band_px <= p.y() <= self.height() + 4

    def _start_resize(self, global_pos):
        self._resizing_height = True
        self._resize_start_y = int(global_pos.y())
        self._resize_start_h = self.height()
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def _continue_resize(self, global_pos):
        dy = int(global_pos.y()) - self._resize_start_y
        new_h = max(self._min_h, min(self._max_h, self._resize_start_h + dy))
        self.setFixedHeight(new_h)
        self.updateGeometry()

    def _end_resize(self):
        self._resizing_height = False
        self.unsetCursor()

    def eventFilter(self, obj, event):
        if obj is self.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton and self._near_bottom_edge(event.globalPosition()):
                    self._start_resize(event.globalPosition())
                    return True

            if event.type() == QEvent.Type.MouseMove:
                if self._resizing_height:
                    self._continue_resize(event.globalPosition())
                    return True

                if self._near_bottom_edge(event.globalPosition()):
                    self.viewport().setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.viewport().unsetCursor()

            if event.type() == QEvent.Type.MouseButtonRelease:
                if self._resizing_height:
                    self._end_resize()
                    self.viewport().unsetCursor()
                    return True

        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._near_bottom_edge(event.globalPosition()):
            self._start_resize(event.globalPosition())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing_height:
            self._continue_resize(event.globalPosition())
            event.accept()
            return

        if self._near_bottom_edge(event.globalPosition()):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            self.unsetCursor()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing_height:
            self._end_resize()
            event.accept()
            return
        super().mouseReleaseEvent(event)

class ResizableTableBox(QWidget):
    """
    Wraps a QTableWidget with a draggable bottom handle that resizes
    the whole table area without interfering with row/column resizing.
    """
    def __init__(self, table: QTableWidget, min_h: int = 120, max_h: int = 500):
        super().__init__()
        self.table = table
        self._min_h = min_h
        self._max_h = max_h
        self._resizing = False
        self._start_y = 0
        self._start_h = min_h

        self.handle = QLabel("")
        self.handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.handle.setFixedHeight(16)
        self.handle.setCursor(Qt.CursorShape.SizeVerCursor)
        self.handle.setStyleSheet(
            "QLabel { "
            "background: palette(midlight); "
            "border: 1px solid palette(mid); "
            "font-size: 10px; "
            "}"
        )

        self.table.setMinimumHeight(min_h)
        self.table.setMaximumHeight(max_h)
        self.table.setFixedHeight(min_h)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.table)
        layout.addWidget(self.handle)

        self.handle.mousePressEvent = self._handle_mouse_press
        self.handle.mouseMoveEvent = self._handle_mouse_move
        self.handle.mouseReleaseEvent = self._handle_mouse_release

    def _handle_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._resizing = True
            self._start_y = int(event.globalPosition().y())
            self._start_h = self.table.height()
            event.accept()

    def _handle_mouse_move(self, event):
        if self._resizing:
            dy = int(event.globalPosition().y()) - self._start_y
            new_h = max(self._min_h, min(self._max_h, self._start_h + dy))
            self.table.setFixedHeight(new_h)
            self.updateGeometry()
            event.accept()

    def _handle_mouse_release(self, event):
        if self._resizing:
            self._resizing = False
            event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wing Path Generator alpha")
        self.cfg = AppConfig()
        self.airfoil = naca4("4412")
        self.plot = PlotWidget()
        self.inputs = QWidget()
        self.input_layout = QVBoxLayout(self.inputs)
        self.build_inputs()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.inputs)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll); splitter.addWidget(self.plot)
        splitter.setSizes([420, 900])
        self.setCentralWidget(splitter)
        self.refresh_plot()

    def center_on_bed(self):
        self.update_from_ui()

        try:
            xmin, xmax, ymin, ymax = all_model_bounds(self.cfg, self.airfoil)
            ox, oy = model_origin_xy(self.cfg, self.airfoil)

            local_xmin = xmin - ox
            local_xmax = xmax - ox
            local_ymin = ymin - oy
            local_ymax = ymax - oy

            part_w = local_xmax - local_xmin
            part_h = local_ymax - local_ymin

            target_xmin = (self.cfg.printer.bed_size_x - part_w) / 2.0
            target_ymin = (self.cfg.printer.bed_size_y - part_h) / 2.0

            new_origin_x = target_xmin - local_xmin
            new_origin_y = target_ymin - local_ymin

            self.origin_x.setValue(new_origin_x)
            self.origin_y.setValue(new_origin_y)

            self.update_from_ui()

        except Exception as e:
            QMessageBox.critical(self, "Center on bed failed", str(e))

    def spin(self, value, lo, hi, step=0.1, decimals=3):
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(decimals); s.setSingleStep(step); s.setValue(value)
        s.valueChanged.connect(self.update_from_ui)
        return s

    def intspin(self, value, lo, hi):
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(value); s.valueChanged.connect(self.update_from_ui); return s

    def build_inputs(self):
        self.build_printer_group()
        self.build_airfoil_group()
        self.build_spar_group()
        self.build_rib_group()
        self.build_profile_group()
        self.build_export_group()
        self.input_layout.addStretch(1)

    def group(self, title: str) -> Tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title); form = QFormLayout(box); self.input_layout.addWidget(box); return box, form

    def build_printer_group(self):
        _, f = self.group("Printer")
        p = self.cfg.printer
        self.bed_size_x = self.spin(p.bed_size_x, 10, 2000, 1, 2)
        f.addRow("Bed size X", self.bed_size_x)

        self.bed_size_y = self.spin(p.bed_size_y, 10, 2000, 1, 2)
        f.addRow("Bed size Y", self.bed_size_y)
        self.filament = self.spin(p.filament_diameter, 0.5, 4, 0.05, 3); f.addRow("Filament diameter", self.filament)
        self.nozzle = self.spin(p.nozzle_diameter, 0.1, 2, 0.05, 3); f.addRow("Nozzle diameter", self.nozzle)
        self.line_width = self.spin(p.line_width, 0.1, 3, 0.01, 3); f.addRow("Line width", self.line_width)
        self.layer_height = self.spin(p.layer_height, 0.02, 1.0, 0.01, 3); f.addRow("Layer height", self.layer_height)
        self.flow = self.spin(p.flow_coeff, 0.1, 3.0, 0.01, 3); f.addRow("Flow coefficient", self.flow)
        self.shells = QComboBox(); self.shells.addItems(["1", "2"]); self.shells.currentIndexChanged.connect(self.update_from_ui); f.addRow("Shell count", self.shells)
        self.inner_speed = self.spin(p.inner_speed, 1, 300, 1, 1); f.addRow("Inner path speed", self.inner_speed)
        self.outer_speed = self.spin(p.outer_speed, 1, 300, 1, 1); f.addRow("Outer skin speed", self.outer_speed)
        self.travel_speed = self.spin(p.travel_speed, 1, 500, 1, 1); f.addRow("Travel speed", self.travel_speed)
        self.retract = self.spin(p.retract_distance, 0, 20, 0.1, 3); f.addRow("Retraction distance", self.retract)
        self.retract_speed = self.spin(p.retract_speed, 1, 200, 1, 1); f.addRow("Retraction speed", self.retract_speed)
        self.clearance = self.spin(p.clearance_travel, 0, 50, 0.5, 2); f.addRow("Top/bottom clearance", self.clearance)
        self.nozzle_temp = self.intspin(p.nozzle_temp, 0, 350); f.addRow("Nozzle temp", self.nozzle_temp)
        self.bed_temp = self.intspin(p.bed_temp, 0, 150); f.addRow("Bed temp", self.bed_temp)
        self.fan = self.intspin(p.fan_speed, 0, 100); f.addRow("Fan %", self.fan)

    def build_airfoil_group(self):
        _, f = self.group("Airfoil & Section")
        self.source_type = QComboBox(); self.source_type.addItems(["NACA 4-digit", "Local .dat", "URL .dat"]); self.source_type.currentIndexChanged.connect(self.update_from_ui)
        self.source = QLineEdit(self.cfg.airfoil.source); self.source.textChanged.connect(self.update_from_ui)
        load = QPushButton("Load airfoil"); load.clicked.connect(self.load_airfoil_clicked)
        browse = QPushButton("Browse"); browse.clicked.connect(self.browse_airfoil)
        row = QHBoxLayout(); row.addWidget(self.source); row.addWidget(browse); row.addWidget(load)
        wrap = QWidget(); wrap.setLayout(row)
        f.addRow("Source type", self.source_type)
        f.addRow("Source", wrap)

    def build_spar_group(self):
        box = QGroupBox("Spars")
        layout = QVBoxLayout(box)
        add = QPushButton("Add spar")
        add.clicked.connect(self.add_spar_row)
        layout.addWidget(add)
        self.spar_table = QTableWidget(0, 5)
        self.spar_table.setHorizontalHeaderLabels(["On", "Placement", "Value", "Diameter", "Delete"])
        self.spar_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(ResizableTableBox(self.spar_table, min_h=125, max_h=420))
        self.input_layout.addWidget(box)
        self.rebuild_spar_table()

    def build_rib_group(self):
        _, f = self.group("Ribbing")
        r = self.cfg.ribs
        self.ribs_enabled = QCheckBox(); self.ribs_enabled.setChecked(r.enabled); self.ribs_enabled.stateChanged.connect(self.update_from_ui); f.addRow("Enable ribs", self.ribs_enabled)
        self.rib_spacing = self.spin(r.spacing, 0.1, 200, 0.5, 3); f.addRow("Rib spacing", self.rib_spacing)
        self.rib_angle = self.spin(r.angle_deg, 5, 85, 1, 2); f.addRow("Rib angle", self.rib_angle)
        self.family_a = QCheckBox(); self.family_a.setChecked(r.family_a); self.family_a.stateChanged.connect(self.update_from_ui); f.addRow("Family A", self.family_a)
        self.family_b = QCheckBox(); self.family_b.setChecked(r.family_b); self.family_b.stateChanged.connect(self.update_from_ui); f.addRow("Family B", self.family_b)

    def build_profile_group(self):
        box = QGroupBox("Wing Profile")
        layout = QVBoxLayout(box)
        add = QPushButton("Add section")
        add.clicked.connect(self.add_station_row)
        layout.addWidget(add)
        self.station_table = QTableWidget(0, 3)
        self.station_table.setHorizontalHeaderLabels(["Span/Z", "Chord", "Delete"])
        self.station_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(ResizableTableBox(self.station_table, min_h=125, max_h=420))
        self.input_layout.addWidget(box)
        self.rebuild_station_table()

    def build_export_group(self):
        box = QGroupBox("Save / Load / Export")
        f = QFormLayout(box)
        center = QPushButton("Center on bed")
        center.clicked.connect(self.center_on_bed)
        f.addRow(center)
        self.origin_x = self.spin(self.cfg.gcode.origin_x, -1000, 1000, 1, 2); f.addRow("Origin X", self.origin_x)
        self.origin_y = self.spin(self.cfg.gcode.origin_y, -1000, 1000, 1, 2); f.addRow("Origin Y", self.origin_y)
        self.z_offset = self.spin(self.cfg.gcode.z_offset, 0, 10, 0.05, 3); f.addRow("Z offset", self.z_offset)
        self.output_file = QLineEdit(self.cfg.gcode.output_file); self.output_file.textChanged.connect(self.update_from_ui); f.addRow("Output file", self.output_file)
        row = QHBoxLayout()
        buttons = [
            ("Save setup", self.save_setup),
            ("Load setup", self.load_setup),
            ("Export G-code", self.export_gcode),
            ("Export X-mirrored", self.export_gcode_mirror_x),
        ]

        for text, func in buttons:
            b = QPushButton(text)
            b.clicked.connect(func)
            row.addWidget(b)
        wrap = QWidget(); wrap.setLayout(row); f.addRow(wrap)
        self.input_layout.addWidget(box)

    def block_table_signals(self, block: bool):
        self.spar_table.blockSignals(block); self.station_table.blockSignals(block)

    def rebuild_spar_table(self):
        self.spar_table.setRowCount(0)
        for sp in self.cfg.spars:
            row = self.spar_table.rowCount(); self.spar_table.insertRow(row)
            on = QCheckBox(); on.setChecked(sp.enabled); on.stateChanged.connect(self.update_from_ui)
            mode = QComboBox(); mode.addItems(["Chord %", "LE distance mm"]); mode.setCurrentText(sp.placement_mode); mode.currentIndexChanged.connect(self.update_from_ui)
            val = QDoubleSpinBox(); val.setRange(-9999, 9999); val.setDecimals(3); val.setValue(sp.placement_value); val.valueChanged.connect(self.update_from_ui)
            dia = QDoubleSpinBox(); dia.setRange(0, 999); dia.setDecimals(3); dia.setValue(sp.diameter); dia.valueChanged.connect(self.update_from_ui)
            dele = QPushButton("Delete"); dele.clicked.connect(lambda _=False, r=row: self.delete_spar_row(r))
            self.spar_table.setCellWidget(row, 0, on); self.spar_table.setCellWidget(row, 1, mode); self.spar_table.setCellWidget(row, 2, val); self.spar_table.setCellWidget(row, 3, dia); self.spar_table.setCellWidget(row, 4, dele)

    def rebuild_station_table(self):
        self.station_table.setRowCount(0)
        for st in self.cfg.stations:
            row = self.station_table.rowCount(); self.station_table.insertRow(row)
            z = QDoubleSpinBox(); z.setRange(0, 100000); z.setDecimals(3); z.setValue(st.z); z.valueChanged.connect(self.update_from_ui)
            chord = QDoubleSpinBox(); chord.setRange(1, 100000); chord.setDecimals(3); chord.setValue(st.chord); chord.valueChanged.connect(self.update_from_ui)
            dele = QPushButton("Delete"); dele.clicked.connect(lambda _=False, r=row: self.delete_station_row(r))
            self.station_table.setCellWidget(row, 0, z); self.station_table.setCellWidget(row, 1, chord); self.station_table.setCellWidget(row, 2, dele)

    def read_table_widgets(self):
        spars = []
        for r in range(self.spar_table.rowCount()):
            spars.append(SparConfig(
                enabled=self.spar_table.cellWidget(r,0).isChecked(),
                placement_mode=self.spar_table.cellWidget(r,1).currentText(),
                placement_value=self.spar_table.cellWidget(r,2).value(),
                diameter=self.spar_table.cellWidget(r,3).value(),
            ))
        stations = []
        for r in range(self.station_table.rowCount()):
            stations.append(StationConfig(
                z=self.station_table.cellWidget(r,0).value(),
                chord=self.station_table.cellWidget(r,1).value(),
            ))
        self.cfg.spars = spars
        self.cfg.stations = sorted(stations, key=lambda s: s.z)

    def update_from_ui(self):
        try:
            p = self.cfg.printer
            p.bed_size_x = self.bed_size_x.value()
            p.bed_size_y = self.bed_size_y.value()
            p.filament_diameter = self.filament.value(); p.nozzle_diameter = self.nozzle.value(); p.line_width = self.line_width.value(); p.layer_height = self.layer_height.value(); p.flow_coeff = self.flow.value(); p.shell_count = int(self.shells.currentText())
            p.inner_speed = self.inner_speed.value(); p.outer_speed = self.outer_speed.value(); p.travel_speed = self.travel_speed.value(); p.retract_distance = self.retract.value(); p.retract_speed = self.retract_speed.value(); p.clearance_travel = self.clearance.value()
            p.nozzle_temp = self.nozzle_temp.value(); p.bed_temp = self.bed_temp.value(); p.fan_speed = self.fan.value()
            self.cfg.airfoil.source_type = self.source_type.currentText(); self.cfg.airfoil.source = self.source.text().strip()
            r = self.cfg.ribs
            r.enabled = self.ribs_enabled.isChecked(); r.spacing = self.rib_spacing.value(); r.angle_deg = self.rib_angle.value(); r.family_a = self.family_a.isChecked(); r.family_b = self.family_b.isChecked()
            self.cfg.gcode.origin_x = self.origin_x.value(); self.cfg.gcode.origin_y = self.origin_y.value(); self.cfg.gcode.z_offset = self.z_offset.value(); self.cfg.gcode.output_file = self.output_file.text().strip() or "wing.gcode"
            self.read_table_widgets()
            self.refresh_plot()
        except Exception as e:
            print("update_from_ui:", e)

    def refresh_plot(self):
        self.plot.set_model(self.cfg, self.airfoil)

    def load_airfoil_clicked(self):
        self.update_from_ui()
        try:
            self.airfoil = load_airfoil(self.cfg.airfoil)
            self.refresh_plot()
        except Exception as e:
            QMessageBox.critical(self, "Airfoil load failed", str(e))

    def browse_airfoil(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open airfoil .dat", "", "Airfoil DAT (*.dat *.txt);;All files (*)")
        if path:
            self.source_type.setCurrentText("Local .dat")
            self.source.setText(path)

    def add_spar_row(self):
        self.read_table_widgets(); self.cfg.spars.append(SparConfig()); self.rebuild_spar_table(); self.update_from_ui()

    def delete_spar_row(self, row: int):
        self.read_table_widgets()
        if 0 <= row < len(self.cfg.spars):
            self.cfg.spars.pop(row)
        self.rebuild_spar_table(); self.update_from_ui()

    def add_station_row(self):
        self.read_table_widgets()
        z = total_span(self.cfg.stations) + 100.0
        chord = self.cfg.stations[-1].chord if self.cfg.stations else 100.0
        self.cfg.stations.append(StationConfig(z, chord))
        self.rebuild_station_table(); self.update_from_ui()

    def delete_station_row(self, row: int):
        self.read_table_widgets()
        if len(self.cfg.stations) <= 1:
            QMessageBox.warning(self, "Cannot delete", "At least one profile station is required.")
            return
        if 0 <= row < len(self.cfg.stations):
            self.cfg.stations.pop(row)
        self.rebuild_station_table(); self.update_from_ui()

    def config_to_dict(self):
        self.update_from_ui()
        return asdict(self.cfg)

    def apply_config_dict(self, d):
        self.cfg = AppConfig(
            printer=PrinterConfig(**d.get("printer", {})),
            airfoil=AirfoilConfig(**d.get("airfoil", {})),
            spars=[SparConfig(**x) for x in d.get("spars", [])],
            ribs=RibConfig(**d.get("ribs", {})),
            stations=[StationConfig(**x) for x in d.get("stations", [])],
            gcode=GCodeConfig(**d.get("gcode", {})),
        )

        # Fallbacks if old/broken JSON has empty lists
        if not self.cfg.spars:
            self.cfg.spars = [SparConfig()]
        if not self.cfg.stations:
            self.cfg.stations = [StationConfig()]

        widgets_to_block = [
            self.bed_size_x,
            self.bed_size_y,
            self.filament,
            self.nozzle,
            self.line_width,
            self.layer_height,
            self.flow,
            self.shells,
            self.inner_speed,
            self.outer_speed,
            self.travel_speed,
            self.retract,
            self.retract_speed,
            self.clearance,
            self.nozzle_temp,
            self.bed_temp,
            self.fan,
            self.source_type,
            self.source,
            self.ribs_enabled,
            self.rib_spacing,
            self.rib_angle,
            self.family_a,
            self.family_b,
            self.origin_x,
            self.origin_y,
            self.z_offset,
            self.output_file,
            self.spar_table,
            self.station_table,
        ]

        for w in widgets_to_block:
            w.blockSignals(True)

        try:
            p = self.cfg.printer
            self.bed_size_x.setValue(p.bed_size_x)
            self.bed_size_y.setValue(p.bed_size_y)
            self.filament.setValue(p.filament_diameter)
            self.nozzle.setValue(p.nozzle_diameter)
            self.line_width.setValue(p.line_width)
            self.layer_height.setValue(p.layer_height)
            self.flow.setValue(p.flow_coeff)
            self.shells.setCurrentText(str(p.shell_count))
            self.inner_speed.setValue(p.inner_speed)
            self.outer_speed.setValue(p.outer_speed)
            self.travel_speed.setValue(p.travel_speed)
            self.retract.setValue(p.retract_distance)
            self.retract_speed.setValue(p.retract_speed)
            self.clearance.setValue(p.clearance_travel)
            self.nozzle_temp.setValue(p.nozzle_temp)
            self.bed_temp.setValue(p.bed_temp)
            self.fan.setValue(p.fan_speed)

            a = self.cfg.airfoil
            self.source_type.setCurrentText(a.source_type)
            self.source.setText(a.source)

            r = self.cfg.ribs
            self.ribs_enabled.setChecked(r.enabled)
            self.rib_spacing.setValue(r.spacing)
            self.rib_angle.setValue(r.angle_deg)
            self.family_a.setChecked(r.family_a)
            self.family_b.setChecked(r.family_b)

            g = self.cfg.gcode
            self.origin_x.setValue(g.origin_x)
            self.origin_y.setValue(g.origin_y)
            self.z_offset.setValue(g.z_offset)
            self.output_file.setText(g.output_file)

            self.rebuild_spar_table()
            self.rebuild_station_table()

        finally:
            for w in widgets_to_block:
                w.blockSignals(False)

        try:
            self.airfoil = load_airfoil(self.cfg.airfoil)
        except Exception as e:
            QMessageBox.warning(self, "Airfoil load failed", str(e))

        self.refresh_plot()

    def save_setup(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save setup", "wing_setup.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config_to_dict(), f, indent=2)

    def load_setup(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load wing setup",
            "",
            "Wing setup JSON (*.json);;All files (*)"
        )

        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)

            self.apply_config_dict(d)

        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    def export_gcode(self, mirror_x: bool = False):
        self.update_from_ui()
        suggested = self.cfg.gcode.output_file or "wing.gcode"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export G-code",
            suggested,
            "G-code (*.gcode *.gc *.g);;All files (*)"
        )
        if not path:
            return

        p = self.cfg.printer
        span = total_span(self.cfg.stations)
        layers = max(1, int(math.ceil(span / max(p.layer_height, 1e-9))))

        progress = QProgressDialog("Exporting G-code...", "Cancel", 0, layers, self)
        progress.setWindowTitle("Export G-code")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def progress_callback(layer_index: int, max_layer: int) -> bool:
            progress.setMaximum(max_layer)
            progress.setValue(layer_index)
            progress.setLabelText(f"Exporting layer {layer_index} of {max_layer}...")
            QApplication.processEvents()
            return not progress.wasCanceled()

        try:
            text = GCodeWriter(
                self.cfg,
                self.airfoil,
                progress_callback=progress_callback,
                mirror_x=mirror_x,
            ).write()

            if progress.wasCanceled():
                QMessageBox.information(self, "Export canceled", "G-code export was canceled.")
                return

            progress.setValue(layers)

            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

            QMessageBox.information(self, "Export complete", f"Wrote {path}")

        except RuntimeError as e:
            if "canceled" in str(e).lower():
                QMessageBox.information(self, "Export canceled", "G-code export was canceled.")
            else:
                QMessageBox.critical(self, "Export failed", str(e))

        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

        finally:
            progress.close()

    def export_gcode_mirror_x(self):
        self.export_gcode(mirror_x=True)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1320, 820)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
