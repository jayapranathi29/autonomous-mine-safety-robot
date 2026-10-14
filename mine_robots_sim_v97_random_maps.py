#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v97 (RANDOM MAPS): every run builds a brand-new mine - gallery dividers + doors, hanging / corridor walls, the four
    hazard patches, the four roof-inspection zones, gas plume, floor terrain, cracks and random obstacles.  A layout is
    only accepted if every scan point is reachable from the base.  --env-seed N repeats a mine, --map classic keeps
    the fixed MATLAB-style mine.
MINE SAFETY ROBOTS  -  Quadruped (LiDAR) + Scout Drone  -  VPython simulation  (v2)
===========================================================================
Install :  pip install vpython numpy matplotlib
Run     :  python mine_robots_sim.py            (opens in your browser)
Options :  --speed 6  --seed 7  --dataset mine_dataset_v2.csv  --outdir mine_sim_output  --show  --regen

Keys (click on the 3D view first)
    k = kill the quadruped        d = kill the drone
    g = show / hide hazard patches + outlines
    + / - = faster / slower       space = pause
    r = drop a ROCK-FALL next to the robots (dynamic obstacle)

v96 (DYNAMIC ENVIRONMENT): everything of v95 is kept; what is new
    * The DRONE now has its own 2-D LiDAR (180 beams, 4 m) and its own occupancy map. It no longer reads the
      true wall map: it plans A* on what it has sensed, re-plans when the path is blocked, slows down /
      side-steps / is repelled by close obstacles (reactive layer) and has a stuck-watchdog.
    * Every run builds a DIFFERENT mine: random obstacles (rock piles, parked machines, fallen timber) and
      ROCK-FALLS that drop onto the floor DURING the mission (reachability of all zones is always guaranteed).
      --env-seed N repeats a layout, --obstacles / --rockfalls change how many, key  r  drops a rock-fall now.
    * --drone-prior blueprint (default) = drone starts with the permanent-wall survey map and discovers the
      new obstacles with LiDAR;  --drone-prior none = drone starts blind and maps everything itself.
    * Fixed v95 bugs that the dynamic flight depends on: drone could never land (hovered at the pad until the
      battery died), low-battery return was routed into the battery-courier states, the courier flew a
      straight line through walls, and the quad waited forever if the drone died.
v45d: roof cracks seen by the drone camera are also highlighted (red) on the mine map.
v45c: drone camera = REAL first-person 3D view of the same mine (roof cracks match the readings); info board in its own canvas.
v45b: uneven mine floor (ripples, mounds, rocks; --terrain 0 = flat, 2 = rough).
v45: BIG live drone camera + HTML information board (right) + 9 live graphs (below the 3D view).
v4: full-screen white mine canvas, long real-mine galleries, 10 logical sectors, slow controllers,
    quadruped LiDAR + scout drone, continuous telemetry/zone streaming, battery-aware return-to-base.

What changed in v2
    * ENVIRONMENT now copies your MATLAB map: start room (bottom-left), low-oxygen box with a
      wall inside it, the horizontal wall at the top-left, the HIGH TEMP / HIGH DUST block in the
      middle, the gas room (top-right) with its wall + door, and the free standing walls.
      Hazard areas are painted on the floor (orange / grey / black / green) with labels,
      room names, metre ticks, thick solid walls and a near top-down camera.
    * COMMUNICATION is much stronger: lower path/wall loss, higher sensitivity, smart routing
      (direct OR best single relay through the other robot / base), retry-until-ACK with
      store-and-forward, 10 Hz radio scheduler, link up/down monitoring with events,
      and a live link-quality table (RSSI, PDR, route) in the HUD.
    * The dataset file is regenerated automatically because the map changed
      (default name mine_dataset_v2.csv, or use --regen).
"""
import argparse, csv, heapq, math, os, sys, time
from collections import deque
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=2.0)      # sim seconds per wall second (1.5 = calm, 1.0 = real time)
ap.add_argument("--width", type=int, default=0, help="canvas width in px (default: fill the screen)")
ap.add_argument("--height", type=int, default=0, help="canvas height in px (default: fill the screen)")
ap.add_argument("--panel", type=int, default=0, help="right panel / drone camera width in px (default: ~40%% of the screen)")
ap.add_argument("--terrain", type=float, default=1.0, help="uneven mine-floor roughness (0 = flat floor, 1 = default, 2 = very rough)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--dataset", default="mine_dataset_v2.csv")
ap.add_argument("--outdir", default="mine_sim_output")
ap.add_argument("--show", action="store_true", help="open matplotlib windows at the end")
ap.add_argument("--regen", action="store_true", help="force re-generation of the dataset CSV")
ap.add_argument("--env-seed", type=int, default=-1, help="seed of the random obstacle layout (-1 = new layout every run)")
ap.add_argument("--obstacles", type=int, default=8, help="random obstacles present from the start (0 = none)")
ap.add_argument("--rockfalls", type=int, default=3, help="obstacles that fall onto the floor DURING the mission")
ap.add_argument("--drone-prior", choices=("blueprint", "none"), default="blueprint",
                help="drone starts with the permanent-wall map (blueprint) or completely blind (none)")
ap.add_argument("--tmax", type=float, default=8000.0, help="hard stop [sim s]")
ap.add_argument("--map", choices=("dynamic", "classic"), default="dynamic",
                help="dynamic = a NEW random mine (walls, doors, hazards, roof zones) every run; classic = the fixed MATLAB-style mine")
ap.add_argument("--layout-tries", type=int, default=400, help="max random layouts tried until one passes the reachability checks")
ap.add_argument("--save-dataset", action="store_true", help="also write this run's sensor dataset to --dataset (CSV)")
ARGS, _ = ap.parse_known_args()
ENV_SEED = ARGS.env_seed if ARGS.env_seed >= 0 else int(time.time() * 1000) % (2 ** 31)     # drives the WHOLE random mine

from vpython import (canvas, vector, box, cylinder, sphere, curve, ring, label, color,
                     rate, graph, gcurve, wtext, vertex, quad)

RNG = np.random.default_rng(ARGS.seed)

# =====================================================================================
# 1.  CONFIGURATION  (all tunable numbers are here)
# =====================================================================================
DT = 0.05                 # physics step [sim s]
FPS = 30                  # screen frames per wall second
TS = ARGS.speed           # sim seconds per wall second
T_MAX = ARGS.tmax        # hard stop [sim s]

# ---- arena (from the MATLAB map) ---------------------------------------------------
X0, X1, Y0, Y1 = 0.0, 24.0, 0.0, 20.0
RES = 0.20
NX = int(round((X1 - X0) / RES))
NY = int(round((Y1 - Y0) / RES))
NZX, NZY = 5, 2                                # 10 large mine sectors; fine grid below is the 125+ cell scan grid
ZW, ZH = (X1 - X0) / NZX, (Y1 - Y0) / NZY
NZ = NZX * NZY
WALL_H = 1.4                                   # physical wall height (radio / drone clearance)
VIS_WALL_H = 0.85                              # long, solid mine walls
BASE_POS = (1.5, 1.5)                           # base station + drone pad
QUAD_DOCK = (1.5, 1.5)

# ---- quadruped ------------------------------------------------------------------
Q_VNOM, Q_WMAX = 0.34, 1.50
Q_CAP_WH, Q_RESERVE = 8.0, 0.12
Q_P_IDLE, Q_P_WALK, Q_P_TURN = 7.0, 13.0, 2.0
Q_CHG_W = 60.0
Q_P_STANDBY = 2.5
LIDAR_N, LIDAR_R, LIDAR_SIG = 160, 4.0, 0.01

# ---- drone ----------------------------------------------------------------------
ALT = 0.72
D_VMAX, D_AMAX, D_VZMAX = 0.30, 0.55, 0.25
D_CAP_WH, D_RESERVE = 20.0, 0.18
D_P_HOVER, D_P_PAY = 70.0, 6.0
D_CHG_W = 200.0
CAM_HFOV = math.radians(32)
FRONT_FOV = math.radians(70)

# ---- return-to-base risk model ----------------------------------------------------
P_MAX = 0.05
P_GATE = 0.02
SIGMA_FRAC = 0.12
PESSIMISM = 1.10

# ---- radio (v2: stronger link budget) ---------------------------------------------
RSSI_P0 = -32.0         # dBm at 1 m  (was -40)
PATH_N = 2.2            # path-loss exponent (was 3.0, mine corridors guide the wave)
WALL_LOSS = 5.0         # dB per wall (was 11)
SENS_DBM = -92.0        # receiver sensitivity (was -88)
RADIO_TICK = 0.1        # s between radio scheduler runs (was 0.25)
LINK_DOWN, LINK_UP = 0.30, 0.60      # hysteresis thresholds on best-route delivery probability
NODES = ("BASE", "QUAD", "DRONE")

# ---- zone classification --------------------------------------------------------
GAS_C, GAS_U = 8.0, 20.0
O2_C, O2_U = 20.5, 19.5
TMP_C, TMP_U = 35.0, 45.0
DST_C, DST_U = 3.0, 10.0
NMIN_Q, NMIN_D = 6, 10
CLS_NAME = {-1: "UNKNOWN", 0: "SAFE", 1: "CAUTION", 2: "UNSAFE"}

# hazard rectangles (x_min, x_max, y_min, y_max) - read from the MATLAB picture
HAZ = {"low_oxygen": (3.0, 7.0, 5.8, 9.0), "high_temp": (10.0, 14.0, 15.2, 19.0),
       "high_dust": (10.0, 14.0, 12.0, 15.2), "high_gas": (18.0, 23.0, 15.5, 19.0)}
DS_T, DS_SLICE_S = 20, 30.0

# ---- inspection architecture ----------------------------------------------------
# Ground quadruped = main environmental inspection robot.
# Drone = structural / roof / crack inspection + battery courier.
SERVICE_PACK_WH = 7.0
SERVICE_TRANSFER_MAX_WH = 6.0
SERVICE_MIN_QUAD_SOC = 0.30
SERVICE_DOCK_ALT = 1.0
SERVICE_DWELL_S = 2.0

# Structural zones are inspected only by the drone camera.
STRUCT_ZONES = {
    "crack_A": (2.0, 6.0, 15.7, 19.0),
    "crack_B": (15.0, 18.8, 10.3, 14.5),
    "roof_stress": (6.8, 10.5, 5.1, 9.2),
    "collapse_risk": (19.0, 23.0, 2.9, 6.8),
}
STRUCT_ORDER = tuple(STRUCT_ZONES.keys())
STRUCT_DISPLAY = {
    "crack_A": "CRACK / ROOF A",
    "crack_B": "CRACK / ROOF B",
    "roof_stress": "ROOF STRESS",
    "collapse_risk": "COLLAPSE RISK",
}
STRUCT_SCAN_FRAC = 0.78
STRUCT_CAUTION = 0.35
STRUCT_DANGER = 0.65

# Fine LiDAR settings: high angular density + low noise.  LiDAR is the
# quadruped's primary local obstacle sensor, not just a visual effect.
LIDAR_N, LIDAR_R, LIDAR_SIG = 240, 5.0, 0.005
LIDAR_UPDATE_S = 0.05
LIDAR_FRONT_HALF_ANGLE = math.radians(32)
LIDAR_STOP_DIST = 0.28
LIDAR_SLOW_DIST = 0.75
LIDAR_OCC_HIT = 0.65
LIDAR_FREE_STEP = 0.10

# ---- drone LiDAR + dynamic navigation (v96) ---------------------------------------------
DL_N, DL_R, DL_SIG = 180, 4.0, 0.005       # beams, range [m], range noise [m]
DL_UPDATE_S = 0.10                          # LiDAR period [s]
D_CLEAR_STOP, D_CLEAR_SLOW = 0.30, 0.90     # clearance in the flight direction: stop / start slowing [m]
D_REPEL_R, D_REPEL_GAIN, D_SIDE = 0.55, 0.30, 0.16   # repulsion radius [m], gain [m/s], side-step speed [m/s]
ZMAX = max(ALT, SERVICE_DOCK_ALT)           # flight ceiling (the battery hand-over is done at SERVICE_DOCK_ALT)


# =====================================================================================
# 2.  GEOMETRY, ZONES, RAY-CASTING
# =====================================================================================
def rect_segs(a, b, c, d):
    return [(a, b, c, b), (c, b, c, d), (c, d, a, d), (a, d, a, b)]

WALLS = rect_segs(X0, Y0, X1, Y1)
# Long mine galleries and cross-passages.  These are deliberately long, solid
# structures rather than small laboratory obstacles.  Gaps are the tunnel doors.
WALLS += [
    # long horizontal gallery dividers
    (1.0, 4.8, 22.8, 4.8), (1.0, 10.0, 22.8, 10.0), (1.0, 15.2, 22.8, 15.2),
    # long vertical side galleries / pillars
    (6.5, 0.8, 6.5, 4.8), (17.8, 4.8, 17.8, 10.0),
    (8.5, 10.0, 8.5, 15.2), (19.5, 15.2, 19.5, 19.2),
    # additional parallel walls creating realistic long tunnel corridors
    (2.0, 2.8, 12.5, 2.8), (15.0, 2.8, 22.0, 2.8),
    (2.0, 7.0, 5.2, 7.0), (7.8, 7.0, 16.0, 7.0), (18.5, 7.0, 22.0, 7.0),
    (2.0, 12.5, 7.0, 12.5), (10.0, 12.5, 18.0, 12.5), (20.5, 12.5, 22.0, 12.5),
    # branch walls and chambers
    (12.0, 0.8, 12.0, 2.8), (20.5, 2.8, 20.5, 4.8),
    (3.5, 15.2, 3.5, 19.2), (14.5, 15.2, 14.5, 19.2),
    (5.0, 17.5, 9.0, 17.5), (9.0, 17.5, 9.0, 19.2),
    (14.5, 17.0, 18.0, 17.0),
]
# Doors / openings are intentionally wide so both robots can navigate safely.
# Convert the wall list to the ray-cast array used by LiDAR/radio.
SEG = np.array(WALLS, dtype=float)
SEG_ALL = SEG.copy()                 # walls + obstacles (ray-casting)   [v96]
OBS_SEG = np.zeros((0, 4))           # obstacle edges only (radio loss)   [v96]
ROOM_LABELS = [("BASE / ENTRY", 1.7, 1.8), ("LONG MAIN GALLERY", 12.0, 1.3),
               ("LOW OXYGEN", 5.0, 6.1), ("HIGH DUST", 12.0, 13.0),
               ("HIGH TEMP", 12.0, 17.4), ("HIGH GAS", 20.5, 17.0)]


def zone_of(x, y):
    zx = min(NZX - 1, max(0, int((x - X0) / ZW)))
    zy = min(NZY - 1, max(0, int((y - Y0) / ZH)))
    return zy * NZX + zx


def zone_center(z):
    return X0 + (z % NZX + 0.5) * ZW, Y0 + (z // NZX + 0.5) * ZH


def cell_of(x, y):
    return (min(NX - 1, max(0, int((x - X0) / RES))), min(NY - 1, max(0, int((y - Y0) / RES))))


def cell_xy(ix, iy):
    return X0 + (ix + 0.5) * RES, Y0 + (iy + 0.5) * RES


def raycast(ox, oy, angles, max_r):
    dx, dy = np.cos(angles), np.sin(angles)
    x1, y1, x2, y2 = SEG_ALL.T
    ex, ey = x2 - x1, y2 - y1
    den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    qx, qy = x1 - ox, y1 - oy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx[None, :] * ey[None, :] - qy[None, :] * ex[None, :]) / den
        u = (qx[None, :] * dy[:, None] - qy[None, :] * dx[:, None]) / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    return np.minimum(t.min(axis=1), max_r)


def _crossings(S, a, b):
    if len(S) == 0:
        return 0
    x1, y1, x2, y2 = S.T
    ax, ay = a
    bx, by = b
    d1 = (bx - ax) * (y1 - ay) - (by - ay) * (x1 - ax)
    d2 = (bx - ax) * (y2 - ay) - (by - ay) * (x2 - ax)
    d3 = (x2 - x1) * (ay - y1) - (y2 - y1) * (ax - x1)
    d4 = (x2 - x1) * (by - y1) - (y2 - y1) * (bx - x1)
    return int(np.sum((d1 * d2 < 0) & (d3 * d4 < 0)))


def walls_crossed(a, b):
    """radio attenuation count: permanent walls count 1, an obstacle (2 crossed edges) counts 1"""
    n = _crossings(SEG, a, b)
    if len(OBS_SEG):
        n += 0.5 * _crossings(OBS_SEG, a, b)
    return n


def dilate(m, n=1):
    out = m.copy()
    for _ in range(n):
        o = out.copy()
        o[1:, :] |= out[:-1, :]; o[:-1, :] |= out[1:, :]
        o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        o[1:, 1:] |= out[:-1, :-1]; o[:-1, :-1] |= out[1:, 1:]
        o[1:, :-1] |= out[:-1, 1:]; o[:-1, 1:] |= out[1:, :-1]
        out = o
    return out


# =====================================================================================
# 2a.  RANDOM MINE GENERATOR (v97): a DIFFERENT mine (walls, doors, hazards, roof zones) every run
# =====================================================================================
MAP_RNG_SEED = ENV_SEED
PLUME_C = (3.7, 3.4)                  # gas-leak centre used by the dataset (the generator moves it next to the gas zone)
_G_NB8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _g_occ(walls):
    m = np.zeros((NY, NX), dtype=bool)
    for (a, b, c, d) in walls:
        n = int(max(abs(c - a), abs(d - b)) / 0.05) + 2
        xs, ys = np.linspace(a, c, n), np.linspace(b, d, n)
        m[np.clip(((ys - Y0) / RES).astype(int), 0, NY - 1), np.clip(((xs - X0) / RES).astype(int), 0, NX - 1)] = True
    return m


def _g_rectmask(rect, margin=0.0):
    a, b, c, d = rect
    ix0, iy0 = cell_of(a - margin, c - margin)
    ix1, iy1 = cell_of(b + margin, d + margin)
    m = np.zeros((NY, NX), dtype=bool)
    m[iy0:iy1 + 1, ix0:ix1 + 1] = True
    return m


def _g_reach(blocked, src):
    seen = np.zeros((NY, NX), dtype=bool)
    if blocked[src[1], src[0]]:
        return seen
    dq = deque([src]); seen[src[1], src[0]] = True
    while dq:
        cx, cy = dq.popleft()
        for dx, dy in _G_NB8:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < NX and 0 <= ny_ < NY and not seen[ny_, nx_] and not blocked[ny_, nx_]:
                if dx and dy and (blocked[cy, nx_] or blocked[ny_, cx]):
                    continue
                seen[ny_, nx_] = True; dq.append((nx_, ny_))
    return seen


def _g_gap(r1, r2):
    """distance between two axis-aligned boxes (xmin, xmax, ymin, ymax)"""
    return math.hypot(max(0.0, r2[0] - r1[1], r1[0] - r2[1]), max(0.0, r2[2] - r1[3], r1[2] - r2[3]))


def _g_pts(rect, occ, spacing):
    a, b, c, d = rect
    out = []
    for yy in np.arange(c + spacing / 2, d, spacing):
        for xx in np.arange(a + spacing / 2, b, spacing):
            ix, iy = cell_of(float(xx), float(yy))
            if not occ[iy, ix]:
                out.append((ix, iy))
    return out


def _g_clear_base(b):
    c = cell_of(*BASE_POS)
    b[max(0, c[1] - 2):c[1] + 3, max(0, c[0] - 2):c[0] + 3] = False
    return b


def _g_layout(rng):
    """one random mine: gallery dividers with doors, hanging walls, corridor walls, 4 hazard patches, 4 roof zones"""
    W = rect_segs(X0, Y0, X1, Y1)
    nh = int(rng.choice([2, 3, 3, 4]))
    step = (Y1 - Y0) / (nh + 1)
    dys = [round(float(Y0 + (i + 1) * step + rng.uniform(-0.10, 0.10) * step), 1) for i in range(nh)]
    divs, prev = [], []
    for y in dys:                                                   # --- long gallery dividers with 2-3 wide doors
        doors = []
        nd = int(rng.choice([2, 2, 3]))
        for _ in range(300):
            c = float(rng.uniform(X0 + 2.8, X1 - 2.8))
            if all(abs(c - d) >= 6.0 for d in doors) and (doors or all(abs(c - d) >= 4.0 for d in prev)):
                doors.append(c)                          # first door is offset from the doors below -> winding route
                if len(doors) == nd:
                    break
        if len(doors) < 2:
            return None
        dw = float(rng.uniform(2.2, 3.0))
        x = X0
        for c in sorted(doors):
            if c - dw / 2 > x + 0.05:
                W.append((x, y, c - dw / 2, y))
            x = c + dw / 2
        W.append((x, y, X1, y))
        divs.append((y, doors, dw)); prev = doors
    bounds = [Y0] + dys + [Y1]
    inner = []
    for gi in range(nh + 1):                                        # --- walls inside each gallery
        yb, yt = bounds[gi], bounds[gi + 1]
        Hg = yt - yb
        want = int(rng.integers(1, 3)) if gi == 0 else int(rng.integers(2, 5))
        got = 0
        for _ in range(want * 15):
            if got >= want:
                break
            if rng.random() < 0.6:                                  # wall hanging from a divider (leaves >= 2.4 m)
                Lmax = Hg - 2.4
                if Lmax < 1.6:
                    continue
                L = float(rng.uniform(min(1.6, Lmax), Lmax)); x = float(rng.uniform(X0 + 3.0, X1 - 3.0))
                from_top = rng.random() < 0.5
                att = gi if from_top else gi - 1                    # divider it is attached to (None = outer wall)
                if 0 <= att < nh and from_top is not None:
                    yy, doors, dw = divs[att]
                    if any(abs(x - c) < dw / 2 + 1.6 for c in doors):
                        continue
                w = (x, yt, x, yt - L) if from_top else (x, yb, x, yb + L)
            else:                                                   # free corridor wall
                if Hg < 4.4:
                    continue
                L = float(rng.uniform(3.0, 8.0))
                if yt - 2.2 <= yb + 2.2:
                    continue
                y = float(rng.uniform(yb + 2.2, yt - 2.2))
                x0 = float(rng.uniform(X0 + 2.4, X1 - 2.4 - L)); w = (x0, y, x0 + L, y)
            bb = (min(w[0], w[2]), max(w[0], w[2]), min(w[1], w[3]), max(w[1], w[3]))
            if _g_gap(bb, (BASE_POS[0], BASE_POS[0], BASE_POS[1], BASE_POS[1])) < 3.2:
                continue
            if any(_g_gap(bb, o) < 2.2 for o in inner):
                continue
            inner.append(bb); W.append(w); got += 1
    occ = _g_occ(W)
    bl = _g_clear_base(dilate(occ, 3))                              # strict test: 0.6 m clearance on both sides
    reach = _g_reach(bl, cell_of(*BASE_POS))
    free = ~bl
    if free.sum() < 0.45 * NX * NY or reach[free].sum() < 0.96 * free.sum():
        return None
    # --- hazard patches (floor areas inside the galleries, never touching a wall)
    names = ["low_oxygen", "high_temp", "high_dust", "high_gas"]
    rng.shuffle(names)
    size = {"low_oxygen": (3.2, 4.4, 2.2, 2.8), "high_temp": (3.2, 4.4, 2.2, 2.8),
            "high_dust": (3.2, 4.4, 2.2, 2.8), "high_gas": (3.6, 5.0, 2.4, 3.0)}
    haz, placed = {}, []
    for n in names:
        wlo, whi, hlo, hhi = size[n]
        ok = None
        for _ in range(500):
            w, h = float(rng.uniform(wlo, whi)), float(rng.uniform(hlo, hhi))
            cx, cy = float(rng.uniform(X0 + 1.2 + w / 2, X1 - 1.2 - w / 2)), float(rng.uniform(Y0 + 1.2 + h / 2, Y1 - 1.2 - h / 2))
            r = (round(cx - w / 2, 1), round(cx + w / 2, 1), round(cy - h / 2, 1), round(cy + h / 2, 1))
            if _g_gap(r, (BASE_POS[0], BASE_POS[0], BASE_POS[1], BASE_POS[1])) < 4.5:
                continue
            if (occ & _g_rectmask(r, 0.4)).any():
                continue
            if any(_g_gap(r, o) < 0.9 for o in placed):
                continue
            ok = r; break
        if ok is None:
            return None
        haz[n] = ok; placed.append(ok)
    # --- roof / structural zones (may contain walls)
    struct = {}
    for n in ("crack_A", "crack_B", "roof_stress", "collapse_risk"):
        ok = None
        for _ in range(500):
            w, h = float(rng.uniform(3.2, 4.4)), float(rng.uniform(2.6, 3.6))
            cx, cy = float(rng.uniform(X0 + 1.0 + w / 2, X1 - 1.0 - w / 2)), float(rng.uniform(Y0 + 1.0 + h / 2, Y1 - 1.0 - h / 2))
            r = (round(cx - w / 2, 1), round(cx + w / 2, 1), round(cy - h / 2, 1), round(cy + h / 2, 1))
            if _g_gap(r, (BASE_POS[0], BASE_POS[0], BASE_POS[1], BASE_POS[1])) < 4.0:
                continue
            if any(_g_gap(r, o) < 0.6 for o in placed):
                continue
            if len(_g_pts(r, occ, 0.75)) < 14:
                continue
            ok = r; break
        if ok is None:
            return None
        struct[n] = ok; placed.append(ok)
    # --- reachability: walls only (scan points) and with the hazard patches treated as no-go
    base_c = cell_of(*BASE_POS)
    r_walls = _g_reach(_g_clear_base(dilate(occ, 2)), base_c)
    hz_mask = np.zeros((NY, NX), dtype=bool)
    for r in haz.values():
        hz_mask |= _g_rectmask(r, 0.4)
    bl2 = _g_clear_base(dilate(occ, 2) | hz_mask)
    r_haz = _g_reach(bl2, base_c)
    keep = ~bl2
    if r_haz[keep].sum() < 0.93 * keep.sum():
        return None
    for r in haz.values():
        pts = _g_pts(r, occ, 1.3)
        if len(pts) < 4 or sum(r_walls[iy, ix] for ix, iy in pts) < 0.9 * len(pts):
            return None
    for r in struct.values():
        pts = [(ix, iy) for ix, iy in _g_pts(r, occ, 1.0) if not bl2[iy, ix]]     # points a robot can stand on
        if len(pts) < 8 or sum(r_haz[iy, ix] for ix, iy in pts) < 0.85 * len(pts):
            return None
    g = haz["high_gas"]
    gx, gy = (g[0] + g[1]) / 2, (g[2] + g[3]) / 2
    ux, uy = (X0 + X1) / 2 - gx, (Y0 + Y1) / 2 - gy
    un = math.hypot(ux, uy) or 1.0
    plume = (gx + ux / un * 3.0, gy + uy / un * 3.0)
    return dict(walls=W, haz=haz, struct=struct, plume=plume, nh=nh)


LAYOUT_INFO = "classic MATLAB-style mine"
if ARGS.map == "dynamic":
    _lay, _att = None, -1
    for _att in range(max(1, ARGS.layout_tries)):
        _lay = _g_layout(np.random.default_rng(MAP_RNG_SEED + 1000 * _att + 5))
        if _lay is not None:
            break
    if _lay is None:
        print(f"[map] no valid random layout after {ARGS.layout_tries} tries -> using the classic map")
    else:
        WALLS[:] = _lay["walls"]
        SEG = np.array(WALLS, dtype=float); SEG_ALL = SEG.copy()
        HAZ.clear(); HAZ.update(_lay["haz"])
        STRUCT_ZONES.clear(); STRUCT_ZONES.update(_lay["struct"])
        PLUME_C = _lay["plume"]
        ROOM_LABELS[:] = [("BASE / ENTRY", BASE_POS[0] + 0.2, BASE_POS[1] + 0.3)] + \
                         [(n.replace("_", " ").upper(), (r[0] + r[1]) / 2, (r[2] + r[3]) / 2) for n, r in HAZ.items()]
        LAYOUT_INFO = f"random mine: {_lay['nh'] + 1} galleries, {len(WALLS)} wall segments (attempt {_att + 1})"
print(f"[map] seed {ENV_SEED}: {LAYOUT_INFO}   (repeat this exact mine with  --env-seed {ENV_SEED})")
for _n, _r in HAZ.items():
    print(f"[map]   hazard  {_n:<11s} x {_r[0]:5.1f}..{_r[1]:5.1f}   y {_r[2]:5.1f}..{_r[3]:5.1f}")
for _n, _r in STRUCT_ZONES.items():
    print(f"[map]   roof    {_n:<13s} x {_r[0]:5.1f}..{_r[1]:5.1f}   y {_r[2]:5.1f}..{_r[3]:5.1f}")


TRUE_OCC = np.zeros((NY, NX), dtype=bool)
for (a, b, c, d) in WALLS:
    n = int(max(abs(c - a), abs(d - b)) / 0.02) + 1
    for s in np.linspace(0, 1, n):
        ix, iy = cell_of(a + s * (c - a), b + s * (d - b))
        TRUE_OCC[iy, ix] = True

CELL_ZONE = np.zeros((NY, NX), dtype=int)
for iy in range(NY):
    for ix in range(NX):
        CELL_ZONE[iy, ix] = zone_of(*cell_xy(ix, iy))

# Coarse sensor footprints used for "fully scanned" coverage.  The physical
# simulation still uses the fine 0.20 m occupancy grid.
def _free_xy(x, y):
    c = cell_of(x, y)
    return 0 <= c[0] < NX and 0 <= c[1] < NY and not TRUE_OCC[c[1], c[0]]

def _sample_points_in_rect(rect, spacing=0.60):
    a, b, c, d = rect
    pts = []
    xs = np.arange(a + spacing/2, b, spacing)
    ys = np.arange(c + spacing/2, d, spacing)
    for yy in ys:
        for xx in xs:
            if _free_xy(float(xx), float(yy)):
                pts.append((float(xx), float(yy)))
    return pts

def _scan_points_snake(rect, spacing=0.78):
    a,b,c,d=rect
    pts=[]
    ys=np.arange(c+spacing/2,d,spacing)
    for row,yy in enumerate(ys):
        xs=np.arange(a+spacing/2,b,spacing)
        if row%2: xs=xs[::-1]
        for xx in xs:
            if _free_xy(float(xx),float(yy)):
                pts.append((float(xx),float(yy)))
    return pts

# =====================================================================================
# 2b.  DYNAMIC ENVIRONMENT (v96): new random obstacles every run + rock-falls during the mission
# =====================================================================================
ENV_RNG = np.random.default_rng(ENV_SEED)
STATIC_OCC = TRUE_OCC.copy()          # permanent mine walls only (the "survey map")
OBSTACLES = []                        # obstacles that are active NOW
OBS_H = 1.0                           # obstacle height [m] -> blocks the LiDAR plane of the drone too
_NB8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
OBS_KINDS = ("ROCK PILE", "MACHINE", "FALLEN TIMBER")


def _rect_mask(rect, margin=0.0):
    a, b, c, d = rect
    ix0, iy0 = cell_of(a - margin, c - margin)
    ix1, iy1 = cell_of(b + margin, d + margin)
    m = np.zeros((NY, NX), dtype=bool)
    m[iy0:iy1 + 1, ix0:ix1 + 1] = True
    return m


def _reach(blocked, src):
    """flood fill (8-connected, no corner cutting): boolean map of the cells reachable from src"""
    seen = np.zeros((NY, NX), dtype=bool)
    if blocked[src[1], src[0]]:
        return seen
    dq = deque([src]); seen[src[1], src[0]] = True
    while dq:
        cx, cy = dq.popleft()
        for dx, dy in _NB8:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < NX and 0 <= ny_ < NY and not seen[ny_, nx_] and not blocked[ny_, nx_]:
                if dx and dy and (blocked[cy, nx_] or blocked[ny_, cx]):
                    continue
                seen[ny_, nx_] = True; dq.append((nx_, ny_))
    return seen


def _dock_clear(b):
    c = cell_of(*BASE_POS)
    b[max(0, c[1] - 2):c[1] + 3, max(0, c[0] - 2):c[0] + 3] = False
    return b


# every point that a robot must be able to reach (hazard scan points + structural viewpoints)
_REQ_CELLS = [cell_of(*p) for rect in HAZ.values() for p in _scan_points_snake(rect, 1.30)]
_REQ_CELLS += [cell_of(*p) for rect in STRUCT_ZONES.values() for p in _sample_points_in_rect(rect, 1.0)]
_BASE_REACH = _reach(_dock_clear(dilate(TRUE_OCC, 2)), cell_of(*BASE_POS))


def _layout_ok(occ_new):
    """True if, with these obstacles, every required point is still reachable from the base pad
    with 0.4 m clearance (stricter than the robots need) and no big region got cut off."""
    bl = _dock_clear(dilate(occ_new, 2))
    reach = _reach(bl, cell_of(*BASE_POS))
    for (ix, iy) in _REQ_CELLS:
        if _BASE_REACH[iy, ix] and not bl[iy, ix] and not reach[iy, ix]:
            return False
    keep = _BASE_REACH & ~bl
    return reach[keep].sum() >= 0.98 * keep.sum()


def _candidate(rng, occ, rects, near=None):
    """random obstacle that keeps the mine fully connected (None if nothing fits)"""
    for _ in range(400):
        if near is None:
            cx = float(rng.uniform(X0 + 1.0, X1 - 1.0)); cy = float(rng.uniform(Y0 + 1.0, Y1 - 1.0))
        else:
            ang = float(rng.uniform(0, 2 * math.pi)); rr = float(rng.uniform(1.8, 4.0))
            cx = near[0] + rr * math.cos(ang); cy = near[1] + rr * math.sin(ang)
            if not (X0 + 1.0 < cx < X1 - 1.0 and Y0 + 1.0 < cy < Y1 - 1.0):
                continue
        w = float(rng.uniform(0.5, 1.1)); h = float(rng.uniform(0.5, 1.1))
        rect = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
        if math.hypot(cx - BASE_POS[0], cy - BASE_POS[1]) < 2.4:
            continue
        if (_rect_mask(rect, 0.6) & STATIC_OCC).any():           # stay 0.6 m away from permanent walls
            continue
        if any(not (rect[1] + 0.8 < r[0] or r[1] + 0.8 < rect[0] or rect[3] + 0.8 < r[2] or r[3] + 0.8 < rect[2])
               for r in rects):
            continue
        if _layout_ok(occ | _rect_mask(rect)):
            return dict(cx=cx, cy=cy, w=w, h=h, rect=rect, kind="ROCK PILE", t_on=None)
    return None


def _plan_obstacles(n_init, n_fall):
    occ = TRUE_OCC.copy(); rects = []; init = []; falls = []
    for i in range(n_init + n_fall):
        ob = _candidate(ENV_RNG, occ, rects)
        if ob is None:
            continue
        occ |= _rect_mask(ob["rect"]); rects.append(ob["rect"])
        if i < n_init:
            ob["kind"] = OBS_KINDS[int(ENV_RNG.integers(len(OBS_KINDS)))]; init.append(ob)
        else:
            ob["kind"] = "ROCKFALL"; falls.append(ob)
    for ob, tt in zip(falls, sorted(float(x) for x in ENV_RNG.uniform(100.0, 900.0, len(falls)))):
        ob["t_on"] = tt
    return init, falls


def activate_obstacle(ob):
    """make an obstacle physically real: floor occupancy, LiDAR edges, radio loss"""
    global OBS_SEG, SEG_ALL
    a, b, c, d = ob["rect"]
    TRUE_OCC[_rect_mask(ob["rect"])] = True
    OBS_SEG = np.vstack([OBS_SEG, np.array(rect_segs(a, c, b, d), dtype=float)])
    SEG_ALL = np.vstack([SEG, OBS_SEG])
    OBSTACLES.append(ob)


INIT_OBS, PENDING_FALLS = _plan_obstacles(max(0, ARGS.obstacles), max(0, ARGS.rockfalls))
for _ob in INIT_OBS:
    activate_obstacle(_ob)
print(f"[env] layout seed {ENV_SEED}: {len(INIT_OBS)} obstacles + {len(PENDING_FALLS)} rock-falls scheduled "
      f"(repeat this mine with  --env-seed {ENV_SEED})")

HAZ_SCAN_POINTS = {name: _scan_points_snake(rect, 1.30) for name, rect in HAZ.items()}
# structural sample points are computed ONCE (their index is used by the coverage bookkeeping)
STRUCT_PTS = {name: _sample_points_in_rect(rect, 0.75) for name, rect in STRUCT_ZONES.items()}

def structural_truth(name, x, y, t):
    a, b, c, d = STRUCT_ZONES[name]
    # Smooth synthetic camera/roof model.  This is a simulation model, not
    # experimentally calibrated mine-roof mechanics.
    gx = max(0.0, 1.0 - abs(x - (a+b)/2) / max((b-a)/2, 1e-6))
    gy = max(0.0, 1.0 - abs(y - (c+d)/2) / max((d-c)/2, 1e-6))
    g = max(0.0, gx * gy)
    if name == "collapse_risk":
        g = min(1.0, 0.65 + 0.35*g)
    crack = float(np.clip(0.08 + 0.82*g + 0.05*math.sin(0.7*x + 0.4*t), 0, 1))
    defl = float(max(0.0, 2.0 + 24.0*crack + 1.5*math.sin(x+y)))
    logit = -4.0 + 5.2*crack + 0.065*defl
    pcol = float(1.0 / (1.0 + math.exp(-logit)))
    return crack, defl, pcol

def structural_class(p):
    return 2 if p >= STRUCT_DANGER else (1 if p >= STRUCT_CAUTION else 0)

def structure_wall_points(name):
    # Build a wall-following route from actual mine walls near the structural zone.
    a,b,c,d = STRUCT_ZONES[name]
    cx, cy = (a+b)/2, (c+d)/2
    expanded=(a-1.0,b+1.0,c-1.0,d+1.0)
    candidates=[]
    for wa,wb,wc,wd in WALLS:
        mx,my=(wa+wc)/2,(wb+wd)/2
        if expanded[0] <= mx <= expanded[1] and expanded[2] <= my <= expanded[3]:
            L=math.hypot(wc-wa,wd-wb)
            if L > 1.0:
                for s in np.linspace(0.12,0.88,max(3,int(L/1.5))):
                    px=wa+s*(wc-wa); py=wb+s*(wd-wb)
                    # Offset away from the wall toward the zone interior.
                    dx,dy=cx-px,cy-py
                    dn=math.hypot(dx,dy)
                    if dn>1e-6:
                        px += 0.65*dx/dn; py += 0.65*dy/dn
                    if X0+0.3<px<X1-0.3 and Y0+0.3<py<Y1-0.3:
                        candidates.append((px,py))
    if not candidates:
        candidates=_sample_points_in_rect(STRUCT_ZONES[name],0.8)
    # Remove near-duplicates.
    out=[]
    for p in candidates:
        if not out or min(math.dist(p,q) for q in out)>0.45:
            out.append(p)
    return out

STRUCT_ROUTE_POINTS={name:structure_wall_points(name) for name in STRUCT_ORDER}

def _build_tunnel_patrol():
    raw=[]
    for yy in np.arange(1.6, Y1 - 1.0, 2.3):
        xs=np.arange(X0+1.0,X1-1.0,1.25)
        for xx in xs:
            if _free_xy(float(xx),float(yy)):
                raw.append((float(xx),float(yy)))
    for xx in np.arange(4.4, X1 - 2.0, 2.8):
        ys=np.arange(Y0+1.0,Y1-1.0,1.25)
        for yy in ys:
            if _free_xy(float(xx),float(yy)):
                raw.append((float(xx),float(yy)))
    out=[]
    for p in raw:
        if not out or min(math.dist(p,q) for q in out)>0.9:
            out.append(p)
    out.sort(key=lambda p:(round(p[1],1),p[0]))
    return out

DRONE_TUNNEL_PATROL=_build_tunnel_patrol()

ZONE_WP = []
_clear = ~dilate(TRUE_OCC, 2)
for z in range(NZ):
    cx, cy = zone_center(z)
    ix, iy = cell_of(cx, cy)
    best = None
    for r in range(0, 12):
        for jx in range(max(0, ix - r), min(NX, ix + r + 1)):
            for jy in range(max(0, iy - r), min(NY, iy + r + 1)):
                if _clear[jy, jx] and CELL_ZONE[jy, jx] == z:
                    dd = (jx - ix) ** 2 + (jy - iy) ** 2
                    if best is None or dd < best[0]:
                        best = (dd, jx, jy)
        if best:
            break
    ZONE_WP.append(cell_xy(best[1], best[2]) if best else (cx, cy))

SNAKE = []
for r in range(NZY):
    row = list(range(r * NZX, (r + 1) * NZX))
    SNAKE += row if r % 2 == 0 else row[::-1]

# =====================================================================================
# 3.  DATASET
# =====================================================================================
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rect(X, Y, r, w=0.12):
    a, b, c, d = r
    return _sig((X - a) / w) * _sig((b - X) / w) * _sig((Y - c) / w) * _sig((d - Y) / w)


def classify(g, o, t, d):
    if g >= GAS_U or o <= O2_U or t >= TMP_U or d >= DST_U:
        return 2
    if g >= GAS_C or o <= O2_C or t >= TMP_C or d >= DST_C:
        return 1
    return 0


def build_raw():
    """vectorised sensor dataset for THIS mine: columns t_s, ix, iy, x, y, zone, gas, o2, temp, dust, label, free"""
    rng = np.random.default_rng(123)
    xs = X0 + (np.arange(NX) + 0.5) * RES
    ys = Y0 + (np.arange(NY) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    IX, IY = np.meshgrid(np.arange(NX), np.arange(NY))
    chunks = []
    for k in range(DS_T):
        tm = k * DS_SLICE_S / 60.0
        so, st, sd, sg = (_rect(X, Y, HAZ[n]) for n in ("low_oxygen", "high_temp", "high_dust", "high_gas"))
        plume = np.exp(-((X - PLUME_C[0]) ** 2 + (Y - PLUME_C[1]) ** 2) / (2 * 0.6 ** 2))
        gas = 2.0 + 34 * sg * (1 + 0.04 * tm) + 10 * plume * (1 + 0.05 * tm) + 4 * sd
        o2 = 20.9 - 4.3 * so - 1.2 * sg - 0.4 * plume
        tmp = 27.0 + 24 * st * (1 + 0.02 * tm) + 4 * sd + 3 * sg + 2 * plume
        dst = 0.5 + 14 * sd + 2.5 * st + 1.0 * plume
        lab = np.vectorize(classify)(gas, o2, tmp, dst)
        gas_m = gas + rng.normal(0, 0.6, gas.shape)
        o2_m = o2 + rng.normal(0, 0.03, gas.shape)
        tmp_m = tmp + rng.normal(0, 0.3, gas.shape)
        dst_m = np.maximum(0, dst + rng.normal(0, 0.15, gas.shape))
        chunks.append(np.column_stack([np.full(gas.size, k * DS_SLICE_S), IX.ravel(), IY.ravel(), X.ravel(), Y.ravel(),
                                       CELL_ZONE.ravel() + 1, gas_m.ravel(), o2_m.ravel(), tmp_m.ravel(), dst_m.ravel(),
                                       lab.ravel(), (~STATIC_OCC).ravel().astype(int)]))
    return np.vstack(chunks)


def generate_dataset(path):
    np.savetxt(path, build_raw(), delimiter=",", comments="",
               header="t_s,ix,iy,x_m,y_m,zone,gas_pct_lel,o2_pct,temp_c,dust_mg_m3,label,free",
               fmt=["%.0f", "%d", "%d", "%.3f", "%.3f", "%d", "%.3f", "%.3f", "%.3f", "%.3f", "%d", "%d"])


class Dataset:
    """Loads the CSV and answers  read(x, y, t) -> [gas, o2, temp, dust]  (time-interpolated)."""

    def __init__(self, path):
        if ARGS.map == "dynamic":                     # a new mine -> a new dataset, built in memory (fast)
            raw = build_raw()
            if ARGS.save_dataset:
                generate_dataset(path); print("[dataset] saved", path)
        else:
            if ARGS.regen or not os.path.exists(path):
                print("[dataset] generating", path)
                generate_dataset(path)
            raw = np.loadtxt(path, delimiter=",", skiprows=1)
        self.raw = raw
        T = int(raw[:, 0].max() / DS_SLICE_S) + 1
        self.T = T
        self.v = np.zeros((T, NY, NX, 4))
        self.label = np.zeros((T, NY, NX), dtype=int)
        k = (raw[:, 0] / DS_SLICE_S).astype(int)
        ix, iy = raw[:, 1].astype(int), raw[:, 2].astype(int)
        self.v[k, iy, ix, :] = raw[:, 6:10]
        self.label[k, iy, ix] = raw[:, 10].astype(int)

    def read(self, x, y, t):
        ix, iy = cell_of(x, y)
        f = min(self.T - 1.0, max(0.0, t / DS_SLICE_S))
        k0 = int(f); k1 = min(self.T - 1, k0 + 1); a = f - k0
        return (1 - a) * self.v[k0, iy, ix] + a * self.v[k1, iy, ix]

    def zone_truth(self):
        out = []
        lab = self.label[self.T // 2]
        for z in range(NZ):
            m = (CELL_ZONE == z) & (~STATIC_OCC)
            n = max(1, m.sum())
            fu, fc = (lab[m] == 2).sum() / n, (lab[m] >= 1).sum() / n
            out.append(2 if fu >= 0.25 else (1 if (lab[m] == 2).any() or fc >= 0.25 else 0))
        return out


# =====================================================================================
# 4.  PLANNING:  A*, Dijkstra, line-of-sight shortcut
# =====================================================================================
NB = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
      (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


def nearest_free(blocked, c, rad=6):
    if not blocked[c[1], c[0]]:
        return c
    best = None
    for jx in range(max(0, c[0] - rad), min(NX, c[0] + rad + 1)):
        for jy in range(max(0, c[1] - rad), min(NY, c[1] + rad + 1)):
            if not blocked[jy, jx]:
                d = (jx - c[0]) ** 2 + (jy - c[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, (jx, jy))
    return best[1] if best else None


def astar(blocked, extra, s, g):
    if blocked[g[1], g[0]] or blocked[s[1], s[0]]:
        return None
    openq = [(0.0, 0.0, s)]
    came, gs = {}, {s: 0.0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == g:
            p = [cur]
            while cur in came:
                cur = came[cur]; p.append(cur)
            return p[::-1]
        if gc > gs.get(cur, 1e18):
            continue
        for dx, dy, w in NB:
            nx_, ny_ = cur[0] + dx, cur[1] + dy
            if not (0 <= nx_ < NX and 0 <= ny_ < NY) or blocked[ny_, nx_]:
                continue
            if dx and dy and (blocked[cur[1], nx_] or blocked[ny_, cur[0]]):
                continue
            ng = gc + w * (1.0 + extra[ny_, nx_])
            if ng < gs.get((nx_, ny_), 1e18):
                gs[(nx_, ny_)] = ng; came[(nx_, ny_)] = cur
                hx, hy = abs(g[0] - nx_), abs(g[1] - ny_)
                h = (hx + hy) + (1.414 - 2) * min(hx, hy)
                heapq.heappush(openq, (ng + h, ng, (nx_, ny_)))
    return None


def dijkstra(blocked, src):
    dist = np.full((NY, NX), np.inf)
    if blocked[src[1], src[0]]:
        return dist
    dist[src[1], src[0]] = 0.0
    q = [(0.0, src)]
    while q:
        d, (cx, cy) = heapq.heappop(q)
        if d > dist[cy, cx]:
            continue
        for dx, dy, w in NB:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < NX and 0 <= ny_ < NY and not blocked[ny_, nx_]:
                if dx and dy and (blocked[cy, nx_] or blocked[ny_, cx]):
                    continue
                nd = d + w * RES
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    heapq.heappush(q, (nd, (nx_, ny_)))
    return dist


def los_clear(blocked, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
    for s in np.linspace(0, 1, n + 1):
        ix = int(round(a[0] + s * (b[0] - a[0]))); iy = int(round(a[1] + s * (b[1] - a[1])))
        if blocked[iy, ix]:
            return False
    return True


def shortcut(blocked, cells):
    if len(cells) < 3:
        return cells
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not los_clear(blocked, cells[i], cells[j]):
            j -= 1
        out.append(cells[j]); i = j
    return out


def path_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


# =====================================================================================
# 5.  LIDAR OCCUPANCY GRID (log-odds)
# =====================================================================================
class OccGrid:
    """LiDAR log-odds occupancy map used for local navigation and replanning."""
    def __init__(self):
        self.L = np.zeros((NY, NX), dtype=float)

    def update(self, ox, oy, angles, ranges, max_r=None):
        max_r = LIDAR_R if max_r is None else max_r      # v96: the drone LiDAR has a different range
        # Ray cells are sampled densely so a thin wall is not skipped.
        ts = np.arange(0.0, max_r + 1e-9, LIDAR_FREE_STEP)
        cx = ox + np.outer(np.cos(angles), ts)
        cy = oy + np.outer(np.sin(angles), ts)
        ix = ((cx-X0)/RES).astype(int)
        iy = ((cy-Y0)/RES).astype(int)
        ok = ((ts[None,:] < np.maximum(ranges[:,None]-RES*0.35, 0.0))
              & (ix>=0)&(ix<NX)&(iy>=0)&(iy<NY))
        self.L[iy[ok],ix[ok]] -= 0.22
        hit = ranges < max_r - 0.015
        hx=((ox+ranges*np.cos(angles)-X0)/RES).astype(int)
        hy=((oy+ranges*np.sin(angles)-Y0)/RES).astype(int)
        h=hit&(hx>=0)&(hx<NX)&(hy>=0)&(hy<NY)
        self.L[hy[h],hx[h]] += LIDAR_OCC_HIT
        np.clip(self.L,-5.0,5.0,out=self.L)

    def occ(self):
        return self.L > 1.15

    def strong_occ(self):
        return self.L > 2.0

    def free(self):
        return self.L < -0.45


# =====================================================================================
# 6.  BATTERY and RETURN-TO-BASE MATHEMATICS
# =====================================================================================
def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class Battery:
    def __init__(self, cap_wh, reserve, cells, r_int):
        self.cap, self.e, self.reserve_frac, self.cells, self.r = cap_wh, cap_wh, reserve, cells, r_int
        self.used = 0.0

    @property
    def soc(self):
        return self.e / self.cap

    @property
    def reserve(self):
        return self.reserve_frac * self.cap

    def drain(self, p_w, dt):
        d = min(self.e, p_w * dt / 3600.0)
        self.e -= d; self.used += d

    def charge(self, p_w, dt):
        taper = 1.0 if self.soc < 0.8 else max(0.15, (1.0 - self.soc) / 0.2)
        self.e = min(self.cap, self.e + p_w * taper * dt / 3600.0)

    def voltage(self, p_w):
        v_oc = self.cells * (3.3 + 0.9 * max(self.soc, 0.0) ** 0.7)
        return v_oc - (p_w / max(v_oc, 1e-3)) * self.r


def p_fail(bat, e_need):
    usable = bat.e - bat.reserve
    sigma = max(SIGMA_FRAC * e_need, 0.01 * bat.cap)
    return 1.0 - Phi((usable - e_need) / sigma)


# =====================================================================================
# 7.  ZONE ACCUMULATORS, KNOWLEDGE, RADIO
# =====================================================================================
VAR_NAMES = ("high gas", "low oxygen", "high temperature", "high dust")
VAR_UNITS = ("%LEL", "% O2", "C", "mg/m3")


def var_levels(v):
    """per-variable level 0 normal / 1 caution / 2 danger  (gas, o2, temp, dust)"""
    g, o, t, d = float(v[0]), float(v[1]), float(v[2]), float(v[3])
    return (2 if g >= GAS_U else 1 if g >= GAS_C else 0,
            2 if o <= O2_U else 1 if o <= O2_C else 0,
            2 if t >= TMP_U else 1 if t >= TMP_C else 0,
            2 if d >= DST_U else 1 if d >= DST_C else 0)


class Acc:
    def __init__(self):
        self.n = self.nc = self.nu = 0
        self.s = np.zeros(4)
        self.sent = -1
        self.prev = -1
        self.ucells = set()
        self.score = np.zeros(4)                       # which variable caused the alarms
        self.ext = np.array([-1e9, 1e9, -1e9, -1e9])   # worst value seen (max gas, MIN o2, max temp, max dust)

    def add(self, vals, c, cell):
        self.n += 1; self.nc += (c == 1); self.nu += (c == 2); self.s += vals
        lv = var_levels(vals)
        for i in range(4):
            self.score[i] += lv[i]
        self.ext = np.array([max(self.ext[0], vals[0]), min(self.ext[1], vals[1]),
                             max(self.ext[2], vals[2]), max(self.ext[3], vals[3])])
        if c == 2:
            self.ucells.add(cell)

    def why(self):
        if self.n == 0 or self.score.max() <= 0:
            return ""
        i = int(np.argmax(self.score))
        return f"{VAR_NAMES[i]} ({self.ext[i]:.1f} {VAR_UNITS[i]})"

    def cls(self):
        if self.n == 0:
            return -1
        fu = self.nu / self.n
        if fu >= 0.25 or (self.prev == 2 and fu >= 0.18):
            r = 2
        elif self.nu > 0 or (self.nc + self.nu) / self.n >= 0.25:
            r = 1
        else:
            r = 0
        self.prev = r
        return r


class Knowledge:
    def __init__(self):
        self.src = {"drone": {}, "ground": {}}
        self.cells = {"drone": {}, "ground": {}}
        self.whys = {"drone": {}, "ground": {}}
        self.version = 0

    def merge(self, m):
        self.src[m["src"]][m["zone"]] = m["cls"]
        self.cells[m["src"]][m["zone"]] = set(m.get("cells", ()))
        self.whys[m["src"]][m["zone"]] = m.get("why", "")
        self.version += 1

    def cls(self, z):
        return max(self.src["drone"].get(z, -1), self.src["ground"].get(z, -1))

    def reason(self, z):
        d, g = self.src["drone"].get(z, -1), self.src["ground"].get(z, -1)
        return self.whys["ground" if g >= d else "drone"].get(z, "")

    def unsafe_cells(self):
        out = set()
        for src in self.cells.values():
            for cs in src.values():
                out |= cs
        return out


class Radio:
    """Mesh radio between BASE, QUAD and DRONE.
    * path loss + wall attenuation (the drone flies above the walls -> far fewer walls in the way)
    * every message is retried every RADIO_TICK until delivered (ACK) or its TTL expires
    * route = direct link OR the best single relay (the other robot or the base)
    * messages of a dead sender are lost, but everything already delivered stays at base
    """

    def __init__(self, sim):
        self.sim = sim
        self.pending = []
        self.sent = self.delivered = self.relayed = self.retries = 0
        self.lat = []

    def rssi(self, a, b):
        pa, pb = self.sim.pos3(a), self.sim.pos3(b)
        d = math.dist(pa, pb)
        w = walls_crossed(pa[:2], pb[:2])
        above = (pa[2] > WALL_H * 0.9) or (pb[2] > WALL_H * 0.9)
        return RSSI_P0 - 10 * PATH_N * math.log10(max(d, 0.3)) - WALL_LOSS * w * (0.25 if above else 1.0)

    def pdr(self, a, b):
        return float(_sig((self.rssi(a, b) - SENS_DBM) / 2.5))

    def route(self, a, b):
        """-> (delivery probability per attempt, relay node or None)"""
        best, via = self.pdr(a, b), None
        for r in NODES:
            if r not in (a, b) and self.sim.alive(r):
                p = self.pdr(a, r) * self.pdr(r, b)
                if p > best:
                    best, via = p, r
        return best, via

    def send(self, src, msg, dests, ttl=None):
        self.sent += 1
        self.pending.append({"src": src, "msg": msg, "dests": set(dests), "t0": self.sim.t, "ttl": ttl})

    def step(self):
        t = self.sim.t
        keep = []
        for p in self.pending:
            if p["ttl"] and t - p["t0"] > p["ttl"]:
                continue
            if not self.sim.alive(p["src"]):
                continue
            for d in list(p["dests"]):
                if not self.sim.alive(d):
                    p["dests"].discard(d); continue
                prob, via = self.route(p["src"], d)
                if RNG.random() < prob:
                    self.sim.deliver(d, p["msg"], p["src"], via is not None)
                    p["dests"].discard(d); self.delivered += 1; self.relayed += via is not None
                    self.lat.append(t - p["t0"])
                else:
                    self.retries += 1
            if p["dests"]:
                keep.append(p)
        self.pending = keep[-600:]


# =====================================================================================
# 8.  QUADRUPED  (LiDAR navigation, A*, pure pursuit, risk based return)
# =====================================================================================
class Quad:
    name="QUAD"

    def __init__(self,sim):
        self.sim=sim
        self.x,self.y=QUAD_DOCK
        self.th=math.pi/2
        self.v=self.w=0.0
        self.bat=Battery(Q_CAP_WH,Q_RESERVE,4,0.35)
        self.grid=OccGrid()
        self.know=Knowledge()
        self.acc=[Acc() for _ in range(NZ)]
        self.state="CHARGE"
        self.path=[]; self.goal_z=None; self.dwell=0.0
        self.alive=True
        self.trail=[(self.x,self.y)]
        self.skip={}
        self.e_ret=self.pf=self.margin=0.0
        self.p_now=0.0
        self.t_scan=self.t_samp=self.t_risk=self.t_plan=self.t_dec=self.t_hb=-9.0
        self.last_rng=None
        self.ang=np.linspace(0,2*math.pi,LIDAR_N,endpoint=False)
        self.phase=0.0
        self.distance=0.0
        self.returns=0
        self.e_per_m=(Q_P_IDLE+Q_P_WALK)/Q_VNOM/3600.0*PESSIMISM
        self.scan_name=None
        self.scan_targets=[]
        self.scan_i=0
        self.wait_reason=""
        self.front_min=LIDAR_R
        self.left_clear=LIDAR_R
        self.right_clear=LIDAR_R
        self.lidar_hits=0

    def unsafe_mask(self):
        m=np.zeros((NY,NX),dtype=bool)
        for (ix,iy) in self.know.unsafe_cells():
            m[iy,ix]=True
        return dilate(m,2)

    def mask_explore(self):
        # Use only high-confidence LiDAR walls.  A single noisy hit must not
        # close an entire tunnel.
        b=dilate(self.grid.strong_occ(),1)|self.unsafe_mask()
        return self._exempt(b)

    def mask_return(self):
        b=dilate(self.grid.strong_occ(),1)|self.unsafe_mask()
        return self._exempt(b)

    def _exempt(self,b):
        c=cell_of(self.x,self.y)
        b[max(0,c[1]-1):c[1]+2,max(0,c[0]-1):c[0]+2]=False
        d=cell_of(*QUAD_DOCK)
        b[max(0,d[1]-1):d[1]+2,max(0,d[0]-1):d[0]+2]=False
        return b

    def extra_cost(self):
        e=np.where(self.grid.L>0.0,0.15,0.0)
        e += 0.35*dilate(self.grid.strong_occ(),1)
        return e

    def plan(self,goal_xy,mode="explore"):
        blocked=self.mask_explore() if mode=="explore" else self.mask_return()
        s=cell_of(self.x,self.y); g=nearest_free(blocked,cell_of(*goal_xy))
        if g is None:return None,0.0
        cells=astar(blocked,self.extra_cost(),s,g)
        if not cells:return None,0.0
        cells=shortcut(blocked,cells)
        pts=[cell_xy(*c) for c in cells[1:]] or [goal_xy]
        return pts,path_len([(self.x,self.y)]+pts)

    def update_risk(self,t):
        blocked=self.mask_return()
        dist=dijkstra(blocked,cell_of(*QUAD_DOCK))
        c=cell_of(self.x,self.y)
        d=dist[c[1],c[0]]
        if not np.isfinite(d): d=1.8*math.dist((self.x,self.y),QUAD_DOCK)
        self.ret_dist=d
        self.e_ret=self.e_per_m*d
        self.pf=p_fail(self.bat,self.e_ret)
        self.margin=self.bat.e-self.bat.reserve-self.e_ret

    def sense_chem(self,t):
        v=self.sim.ds.read(self.x,self.y,t)+RNG.normal(0,[0.8,0.05,0.4,0.25])
        z,c=zone_of(self.x,self.y),classify(*v)
        self.acc[z].add(v,c,cell_of(self.x,self.y))
        self.sim.record_hazard_sample(self.x,self.y,v)
        return z,c,v

    def report(self,z,force=False):
        a=self.acc[z]
        if a.n<NMIN_Q and not force:return
        c=a.cls()
        if c==a.sent:return
        a.sent=c
        self.sim.send_zone(self.name,"ground",z,c,(a.s/max(a.n,1)).round(2).tolist(),a.n,a.ucells,a.why())

    def _lidar_control(self,target):
        if self.last_rng is None:
            return 0.0,Q_VNOM
        rel=((self.ang+self.th-self.th+math.pi)%(2*math.pi))-math.pi
        front=np.abs(rel)<LIDAR_FRONT_HALF_ANGLE
        left=(rel>math.radians(35))&(rel<math.radians(115))
        right=(rel<math.radians(-35))&(rel>math.radians(-115))
        self.front_min=float(np.percentile(self.last_rng[front],8)) if np.any(front) else LIDAR_R
        self.left_clear=float(np.percentile(self.last_rng[left],30)) if np.any(left) else LIDAR_R
        self.right_clear=float(np.percentile(self.last_rng[right],30)) if np.any(right) else LIDAR_R
        goal=math.atan2(target[1]-self.y,target[0]-self.x)
        err=(goal-self.th+math.pi)%(2*math.pi)-math.pi
        # VFH-like local steering: when a wall is close, select the more open side
        # while still retaining the A*/goal heading.
        if self.front_min<LIDAR_STOP_DIST:
            side=1.0 if self.left_clear>=self.right_clear else -1.0
            w=side*Q_WMAX
            v=0.018
        elif self.front_min<LIDAR_SLOW_DIST:
            side_bias=0.0
            if self.left_clear<0.55 and self.right_clear>self.left_clear: side_bias=-0.35
            if self.right_clear<0.55 and self.left_clear>self.right_clear: side_bias=0.35
            w=max(-Q_WMAX,min(Q_WMAX,2.4*err+side_bias))
            v=0.055
        else:
            w=max(-Q_WMAX,min(Q_WMAX,3.0*err))
            v=Q_VNOM*max(0.0,math.cos(err))**2
        return w,v

    def blocked_ahead(self):
        if self.last_rng is None:return False
        rel=((self.ang+self.th-self.th+math.pi)%(2*math.pi))-math.pi
        return bool(np.percentile(self.last_rng[np.abs(rel)<math.radians(18)],10)<LIDAR_STOP_DIST)

    def _start_hazard_scan(self,name):
        pts=list(HAZ_SCAN_POINTS.get(name,()))
        if not pts:return False
        # Start with nearest target and then repeatedly choose the nearest remaining
        # point. This creates a practical coverage path instead of a single center goal.
        self.scan_name=name
        self.scan_targets=pts
        self.scan_i=0
        self.state="SCAN"
        self.path=[]
        self._wd_pos=(self.x,self.y);self._wd_t=self.sim.t
        self.sim.event("QUAD",f"starting FULL {name.replace('_',' ').upper()} scan ({len(pts)} cells)")
        return True

    def _next_scan_target(self):
        if not self.scan_targets:return None
        remaining=self.scan_targets[self.scan_i:]
        if not remaining:return None
        target=min(remaining,key=lambda p:math.dist((self.x,self.y),p))
        idx=self.scan_i+remaining.index(target)
        # swap chosen target to current index
        self.scan_targets[self.scan_i],self.scan_targets[idx]=self.scan_targets[idx],self.scan_targets[self.scan_i]
        return self.scan_targets[self.scan_i]

    def step(self,dt,t):
        if not self.alive:return
        moving=self.state in ("GO","SCAN","RETURN")
        p=Q_P_IDLE+(Q_P_WALK*abs(self.v)/Q_VNOM+Q_P_TURN*abs(self.w)/Q_WMAX if moving else 0.0)
        if self.state in ("WAIT","WAIT_SERVICE","DONE","CHARGE"):p=Q_P_STANDBY
        self.p_now=p
        if self.state=="CHARGE":
            self.bat.charge(Q_CHG_W,dt);self.p_now=-Q_CHG_W
        else:
            self.bat.drain(p,dt)
            if self.bat.e<=1e-6:self.sim.kill("QUAD","battery depleted");return

        if t-self.t_scan>=LIDAR_UPDATE_S:
            self.t_scan=t
            world_ang=self.ang+self.th
            r=raycast(self.x,self.y,world_ang,LIDAR_R)+RNG.normal(0,LIDAR_SIG,LIDAR_N)
            self.last_rng=np.clip(r,0.03,LIDAR_R)
            self.grid.update(self.x,self.y,world_ang,self.last_rng)
            self.lidar_hits=int(np.sum(self.last_rng<LIDAR_R-0.03))

        if t-self.t_samp>=0.12 and self.state not in ("CHARGE","WAIT_SERVICE","DONE"):
            self.t_samp=t
            self.sense_chem(t)

        if t-self.t_risk>=0.5:
            self.t_risk=t;self.update_risk(t)
            drone_ok=self.sim.alive("DRONE")
            if self.state not in ("CHARGE","WAIT_SERVICE","DONE") and self.bat.soc<=SERVICE_MIN_QUAD_SOC and drone_ok:
                self.wait_reason="battery service"
                self.state="WAIT_SERVICE";self.path=[]
                self.sim.request_battery_service()
                self.sim.event("QUAD",f"SOC {self.bat.soc*100:.0f}% -> WAIT for DRONE battery delivery")
            elif self.state in ("GO","SCAN") and self.pf>P_MAX and drone_ok:
                # Do not physically return the ground robot. Ask the drone for a pack.
                self.wait_reason="return-risk"
                self.state="WAIT_SERVICE";self.path=[]
                self.sim.request_battery_service()
                self.sim.event("QUAD",f"return-risk {self.pf:.3f} -> hold position and request DRONE service")
            elif self.state in ("GO","SCAN") and self.pf>P_MAX and not drone_ok:
                # v96: nobody can bring a battery any more -> the quadruped walks home by itself
                pts,_=self.plan(QUAD_DOCK,"return");self.path=pts or [QUAD_DOCK];self.state="RETURN"
                self.sim.event("QUAD",f"return-risk {self.pf:.3f} and DRONE lost -> returning to BASE")

        if t-self.t_hb>=0.5:
            self.t_hb=t;self.sim.send_tel(self.name)
        getattr(self,"st_"+self.state.lower(),lambda d,tt:None)(dt,t)
        self.integrate(dt)

    def st_charge(self,dt,t):
        self.v=self.w=0.0
        if self.bat.soc>=0.97:
            self.state="WAIT";self.sim.event("QUAD","charged - ready for environmental scan")

    def st_wait(self,dt,t):
        self.v=self.w=0.0
        if t-self.t_dec<0.8:return
        self.t_dec=t
        # Always finish environmental hazard scans before considering the mission done.
        for name in HAZ:
            if not self.sim.hazard_done[name]:
                if self._start_hazard_scan(name):return
        if self.sim.all_hazards_done():
            if math.dist((self.x,self.y),QUAD_DOCK)<0.3:
                self.state="DONE";self.sim.event("QUAD","all environmental hazards scanned - mission complete")
            else:
                pts,_=self.plan(QUAD_DOCK,"return")
                self.path=pts or [QUAD_DOCK];self.state="RETURN"

    def st_scan(self,dt,t):
        target=self._next_scan_target()
        if target is None:
            self.sim.finish_hazard_scan(self.scan_name)
            self.scan_name=None;self.scan_targets=[];self.scan_i=0;self.state="WAIT";return
        # v97 stuck-watchdog: a scan cell the robot cannot make progress towards (random obstacle, rock-fall) is skipped
        if math.dist((self.x,self.y),self._wd_pos)>0.35:
            self._wd_pos=(self.x,self.y);self._wd_t=t
        elif t-self._wd_t>15.0:
            try:self.sim.hazard_visited[self.scan_name].add(HAZ_SCAN_POINTS[self.scan_name].index(target))
            except ValueError:pass
            self.sim.event("QUAD",f"scan cell ({target[0]:.1f},{target[1]:.1f}) unreachable (obstacle) - skipped")
            self.scan_i+=1;self.path=[];self._wd_t=t;self._wd_pos=(self.x,self.y)
            return
        if not self.path or t-self.t_plan>1.2 or self.blocked_ahead():
            pts,_=self.plan(target,"explore")
            self.path=pts or [target];self.t_plan=t
        if self.follow(dt):
            self.scan_i+=1
            self.path=[]

    def st_wait_service(self,dt,t):
        self.v=self.w=0.0
        if not self.sim.alive("DRONE") and t-self.t_dec>=1.0:     # v96: do not wait forever for a dead drone
            self.t_dec=t;self.sim.service_requested=False
            self.sim.event("QUAD","DRONE lost - no battery delivery possible, returning to BASE on its own")
            pts,_=self.plan(QUAD_DOCK,"return");self.path=pts or [QUAD_DOCK];self.state="RETURN"

    def st_return(self,dt,t):
        if not self.path or t-self.t_plan>1.5 or self.blocked_ahead():
            pts,_=self.plan(QUAD_DOCK,"return")
            self.path=pts or [QUAD_DOCK];self.t_plan=t
        if self.follow(dt):
            self.state="CHARGE";self.returns+=1
            self.sim.event("QUAD",f"docked, SOC {self.bat.soc*100:.0f}%")

    def follow(self,dt):
        while self.path and math.dist((self.x,self.y),self.path[0])<(0.08 if len(self.path)==1 else 0.14):
            self.path.pop(0)
        if not self.path:
            self.v=self.w=0.0;return True
        tx,ty=self.path[0]
        self.w,self.v=self._lidar_control((tx,ty))
        # Close-range approach. LiDAR controls speed/heading; it does not freeze the robot.
        dist=math.dist((self.x,self.y),(tx,ty))
        if len(self.path)==1:self.v=min(self.v,0.8*dist+0.035)
        return False

    def integrate(self,dt):
        if self.state in ("CHARGE","WAIT","WAIT_SERVICE","DONE"):
            self.v=self.w=0.0
        self.th=(self.th+self.w*dt+math.pi)%(2*math.pi)-math.pi
        nx=self.x+self.v*dt*math.cos(self.th)
        ny=self.y+self.v*dt*math.sin(self.th)
        ix,iy=cell_of(nx,ny)
        if TRUE_OCC[iy,ix]:
            # Physical collision guard: turn away, do not kill motion state.
            self.v=0.0
            self.w=Q_WMAX if self.left_clear>=self.right_clear else -Q_WMAX
        else:
            self.distance+=math.dist((self.x,self.y),(nx,ny));self.x,self.y=nx,ny
            if math.dist((self.x,self.y),self.trail[-1])>0.08:self.trail.append((self.x,self.y))
        self.phase+=2*math.pi*self.v*dt/0.22


# =====================================================================================
# 9.  DRONE
# =====================================================================================
class Drone:
    """Scout drone.  v96: DYNAMIC navigation - the drone senses with its own LiDAR, keeps its own occupancy
    map, plans A* on that map, re-plans when something new blocks the route and has a reactive safety layer."""
    name="DRONE"

    def __init__(self,sim):
        self.sim=sim
        self.x,self.y,self.z=BASE_POS[0],BASE_POS[1],0.04
        self.vx=self.vy=self.vz=0.0
        self.yaw=math.pi/2
        self.bat=Battery(D_CAP_WH,D_RESERVE,3,0.05)
        self.know=Knowledge()
        self.acc=[Acc() for _ in range(NZ)]
        self.state="CHARGE";self.alive=True
        self.target=None;self.wps=[];self.dwell=0.0
        self.e_ret=self.pf=self.margin=self.p_now=0.0
        self.t_risk=self.t_hb=-9.0
        self.e_per_m=(D_P_HOVER*1.25+D_P_PAY)/D_VMAX/3600.0*PESSIMISM
        self.sorties=0;self.distance=0.0
        self.struct_index=0;self.struct_points=[];self.struct_point_i=0
        self.struct_name=None
        self.struct_cov=0.0
        self.camera_status="CAMERA READY"
        self.camera_crack=0.0;self.camera_defl=0.0;self.camera_p=0.0
        self.service_pack=0.0
        self.service_msg="No battery service requested"
        # ---- v96: onboard LiDAR, own occupancy map, navigation state ---------------------------
        self.grid=OccGrid()
        if ARGS.drone_prior=="blueprint":
            self.grid.L[STATIC_OCC]=2.6          # pre-loaded survey of the PERMANENT walls only
        self.ang=np.linspace(0,2*math.pi,DL_N,endpoint=False)
        self.last_rng=None;self.last_wang=self.ang.copy()
        self.t_scan=-9.0;self.lidar_hits=0
        self.front_min=self.left_clear=self.right_clear=DL_R
        self.nav_goal=None;self.nav_status="idle";self.nav_fail=0
        self.t_plan=self.t_chk=-9.0;self.stuck_ref=(self.x,self.y,0.0);self.stuck_n=0
        self.replans=self.stuck_total=self.skipped=self.collisions=0
        self.avoiding=False
        self.ret_dist=None;self.t_dij=-9.0

    @property
    def footprint(self):
        return 2*max(self.z,0.2)*math.tan(CAM_HFOV/2)

    # ------------------------------------------------------------------ sensing
    def lidar_scan(self,t):
        if t-self.t_scan<DL_UPDATE_S:return
        self.t_scan=t
        wang=self.ang+self.yaw
        r=raycast(self.x,self.y,wang,DL_R)+RNG.normal(0,DL_SIG,DL_N)
        self.last_rng=np.clip(r,0.03,DL_R);self.last_wang=wang
        self.grid.update(self.x,self.y,wang,self.last_rng,DL_R)
        self.lidar_hits=int(np.sum(self.last_rng<DL_R-0.03))

    # ------------------------------------------------------------------ planning on the drone's OWN map
    def mask(self,clear=1):
        b=self.grid.strong_occ()
        b=dilate(b,clear) if clear>0 else b.copy()
        c=cell_of(self.x,self.y)
        b[max(0,c[1]-1):c[1]+2,max(0,c[0]-1):c[0]+2]=False
        d=cell_of(*BASE_POS)
        b[max(0,d[1]-1):d[1]+2,max(0,d[0]-1):d[0]+2]=False
        return b

    def extra_cost(self):
        s=self.grid.strong_occ()
        return np.where(self.grid.L>0.0,0.15,0.0)+0.35*dilate(s,1)+0.15*dilate(s,3)   # prefers the tunnel centre

    def plan(self,goal):
        """A* on the LiDAR map. Tries 0.4 m clearance first, then 0.2 m, then squeezes through. None = no route."""
        s=cell_of(self.x,self.y);g0=cell_of(*goal)
        extra=self.extra_cost()
        for clear in (2,1,0):
            b=self.mask(clear)
            g=nearest_free(b,g0)
            if g is None:continue
            cells=astar(b,extra,s,g)
            if cells:
                if len(cells)>2:cells=shortcut(b,cells)
                return [cell_xy(*c) for c in cells[1:]] or [cell_xy(*g)]
        return None

    def path_blocked(self):
        """does the route ahead now cross a cell that the LiDAR has confirmed as occupied?"""
        if not self.wps:return False
        b=self.mask(0);p=(self.x,self.y)
        for q in self.wps[:3]:
            if not los_clear(b,cell_of(*p),cell_of(*q)):return True
            p=q
        return False

    def dist_home(self):
        b=self.mask(1)
        dist=dijkstra(b,cell_of(*BASE_POS))
        c=cell_of(self.x,self.y);d=dist[c[1],c[0]]
        straight=math.hypot(self.x-BASE_POS[0],self.y-BASE_POS[1])
        if not np.isfinite(d):d=1.8*straight
        return float(max(d,straight))

    def ret_energy(self,x,y,z):
        d=self.ret_dist if self.ret_dist is not None else math.hypot(x-BASE_POS[0],y-BASE_POS[1])
        return self.e_per_m*(d+z*0.5)+(D_P_HOVER+D_P_PAY)*5/3600.0

    # ------------------------------------------------------------------ reactive safety layer
    def avoid(self,vdx,vdy):
        """slow down in front of obstacles, side-step to the more open side, repel from anything very close"""
        r=self.last_rng
        if r is None:
            self.avoiding=False;return vdx,vdy
        wa=self.last_wang;k=1.0;lat=(0.0,0.0)
        if math.hypot(vdx,vdy)>1e-6:
            hd=math.atan2(vdy,vdx)
            rel=(wa-hd+math.pi)%(2*math.pi)-math.pi
            cone=np.abs(rel)<math.radians(35)
            left=(rel>math.radians(35))&(rel<math.radians(115))
            right=(rel<-math.radians(35))&(rel>-math.radians(115))
            self.front_min=float(np.percentile(r[cone],10)) if cone.any() else DL_R
            self.left_clear=float(np.percentile(r[left],30)) if left.any() else DL_R
            self.right_clear=float(np.percentile(r[right],30)) if right.any() else DL_R
            if self.front_min<D_CLEAR_STOP:k=0.0
            elif self.front_min<D_CLEAR_SLOW:k=(self.front_min-D_CLEAR_STOP)/(D_CLEAR_SLOW-D_CLEAR_STOP)
            if k<1.0:
                side=1.0 if self.left_clear>=self.right_clear else -1.0
                lat=(-math.sin(hd)*side*D_SIDE*(1-k),math.cos(hd)*side*D_SIDE*(1-k))
        near=r<D_REPEL_R
        rx=ry=0.0
        if near.any():
            wgt=(D_REPEL_R-r[near])/D_REPEL_R
            rx=-float(np.sum(np.cos(wa[near])*wgt))/(DL_N/8.0)*D_REPEL_GAIN
            ry=-float(np.sum(np.sin(wa[near])*wgt))/(DL_N/8.0)*D_REPEL_GAIN
            m=math.hypot(rx,ry)
            if m>0.25:rx*=0.25/m;ry*=0.25/m
        self.avoiding=bool(k<0.99 or near.any())
        return vdx*k+lat[0]+rx,vdy*k+lat[1]+ry

    # ------------------------------------------------------------------ low level flight
    def fly_to(self,tx,ty,tz,dt,vmax=D_VMAX,avoid=True):
        ex,ey,ez=tx-self.x,ty-self.y,tz-self.z
        h=math.hypot(ex,ey);vh=min(vmax,1.6*h)
        vdx=ex/h*vh if h>1e-6 else 0.0;vdy=ey/h*vh if h>1e-6 else 0.0
        if avoid and self.z>0.30:vdx,vdy=self.avoid(vdx,vdy)
        else:self.avoiding=False
        vdz=max(-D_VZMAX,min(D_VZMAX,1.8*ez))
        dvx,dvy,dvz=vdx-self.vx,vdy-self.vy,vdz-self.vz
        n=math.sqrt(dvx*dvx+dvy*dvy+dvz*dvz);lim=D_AMAX*dt
        if n>lim:dvx,dvy,dvz=dvx*lim/n,dvy*lim/n,dvz*lim/n
        self.vx+=dvx;self.vy+=dvy;self.vz+=dvz
        self.distance+=math.sqrt(self.vx**2+self.vy**2+self.vz**2)*dt
        nx,ny=self.x+self.vx*dt,self.y+self.vy*dt
        ix,iy=cell_of(nx,ny)
        if self.z>0.10 and TRUE_OCC[iy,ix]:        # physical contact guard: never fly into a wall / obstacle
            self.vx=self.vy=0.0;self.collisions+=1
        else:
            self.x,self.y=nx,ny
        self.z=max(0.04,min(ZMAX,self.z+self.vz*dt))
        sp=math.hypot(self.vx,self.vy)
        if sp>0.04:
            target_yaw=math.atan2(self.vy,self.vx)
            de=(target_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
            self.yaw+=max(-3.5*dt,min(3.5*dt,de))
        return h

    # ------------------------------------------------------------------ dynamic navigation
    def nav_to(self,gx,gy,gz,dt,t,vmax=D_VMAX,tol=0.12,critical=False):
        """fly to (gx,gy) through the free space of the drone's own LiDAR map.
        returns 'arrived' | 'moving' | 'unreachable'.  critical goals (base, quadruped) never give up."""
        goal=(gx,gy)
        if self.nav_goal is None or math.dist(self.nav_goal,goal)>0.30:
            self.nav_goal=goal;self.wps=[];self.t_plan=-9.0;self.nav_fail=0
            self.stuck_n=0;self.stuck_ref=(self.x,self.y,t)
        if math.hypot(gx-self.x,gy-self.y)<=tol:
            self.nav_goal=None;self.wps=[];self.nav_status="arrived";return "arrived"
        blocked_now=False
        if self.wps and t-self.t_chk>=0.3:
            self.t_chk=t;blocked_now=self.path_blocked()
        if (not self.wps) or blocked_now or t-self.t_plan>=1.5:
            pts=self.plan(goal);self.t_plan=t
            if pts:
                self.wps=pts;self.nav_fail=0;self.replans+=1
            else:
                self.wps=[];self.nav_fail+=1
        if not self.wps:                                   # no route on the current map
            if self.nav_fail>=3 and not critical:
                self.nav_goal=None;self.nav_status="unreachable";return "unreachable"
            if critical:self.fly_to(gx,gy,gz,dt,vmax=0.6*vmax)     # creep toward it, reactive layer keeps it safe
            else:self.fly_to(self.x,self.y,gz,dt)                   # hold position and look again
            self.nav_status="no path";return "moving"
        while len(self.wps)>1 and math.hypot(self.wps[0][0]-self.x,self.wps[0][1]-self.y)<0.18:
            self.wps.pop(0)
        wx,wy=self.wps[0]
        h=self.fly_to(wx,wy,gz,dt,vmax)
        if len(self.wps)==1 and (h<tol or math.hypot(gx-self.x,gy-self.y)<0.25):
            self.nav_goal=None;self.wps=[];self.nav_status="arrived";return "arrived"
        if t-self.stuck_ref[2]>=8.0:                       # watchdog: no progress for 8 s
            moved=math.hypot(self.x-self.stuck_ref[0],self.y-self.stuck_ref[1])
            self.stuck_ref=(self.x,self.y,t)
            if moved<0.20:
                self.stuck_n+=1;self.stuck_total+=1;self.wps=[]
                if self.stuck_n>=3 and not critical:
                    self.nav_goal=None;self.nav_status="unreachable";return "unreachable"
            else:self.stuck_n=0
        self.nav_status="avoiding" if self.avoiding else "following"
        return "moving"

    def _go_home(self,dt,t):
        """dynamic flight to the base pad, then a controlled descent. True once landed."""
        if math.hypot(self.x-BASE_POS[0],self.y-BASE_POS[1])>=0.30:
            self.nav_to(BASE_POS[0],BASE_POS[1],ALT,dt,t,critical=True)
            return False
        self.nav_goal=None;self.wps=[]
        self.fly_to(BASE_POS[0],BASE_POS[1],0.04,dt,avoid=False)
        return self.z<=0.08

    # ------------------------------------------------------------------ structural inspection (unchanged logic)
    def todo_struct(self):
        return [n for n in STRUCT_ORDER if not self.sim.struct_done[n]]

    def start_next_struct(self):
        todo=self.todo_struct()
        if not todo:return False
        name=todo[0]
        # Viewpoints: tunnel-centre points near the zone, wall-following points and a 1 m grid inside the zone.
        # (v96: the flight to each viewpoint is planned dynamically on the LiDAR map.)
        pts=[]
        if DRONE_TUNNEL_PATROL:
            cx,cy=(STRUCT_ZONES[name][0]+STRUCT_ZONES[name][1])/2,(STRUCT_ZONES[name][2]+STRUCT_ZONES[name][3])/2
            near=sorted(DRONE_TUNNEL_PATROL,key=lambda p:math.dist((cx,cy),p))[:8]
            pts.extend(near[::2]); pts.extend(near[1::2][::-1])
        pts += list(STRUCT_ROUTE_POINTS.get(name,()))
        pts += _sample_points_in_rect(STRUCT_ZONES[name],1.0)
        if not pts:return False
        ordered=[];cur=(self.x,self.y);remaining=pts[:]
        while remaining:
            p=min(remaining,key=lambda q:math.dist(cur,q));remaining.remove(p)
            ordered.append(p);cur=p
        self.struct_name=name;self.struct_points=ordered;self.struct_point_i=0
        self.wps=[];self.nav_goal=None;self.state="STRUCTURE"
        self.sim.event("DRONE",f"camera inspection -> {STRUCT_DISPLAY[name]} ({len(ordered)} wall/roof viewpoints)")
        return True

    def camera_sample(self,t):
        if self.struct_name is None:
            self.camera_status="TUNNEL TRANSIT"
            self.camera_crack=self.camera_defl=self.camera_p=0.0
            return
        crack,defl,p=structural_truth(self.struct_name,self.x,self.y,t)
        self.camera_crack=crack;self.camera_defl=defl;self.camera_p=p
        if crack>=0.45:
            self.camera_status=f"CRACK DETECTED  {crack:.2f}"
        else:
            self.camera_status=f"NO MAJOR CRACK  {crack:.2f}"

    def record_structure(self,t):
        if self.struct_name is None:return
        self.sim.record_struct_sample(self.struct_name,self.x,self.y,t)

    # ------------------------------------------------------------------ main step
    def step(self,dt,t):
        if not self.alive:return
        airborne=self.state not in ("CHARGE","DONE")
        sp=math.hypot(self.vx,self.vy)
        if airborne:
            self.p_now=D_P_HOVER*(1+0.20*(sp/D_VMAX)**2)+D_P_PAY+35*max(0.0,self.vz)
            self.bat.drain(self.p_now,dt)
            if self.bat.e<=1e-6:self.sim.kill("DRONE","battery depleted");return
            self.lidar_scan(t)
        else:
            self.bat.charge(D_CHG_W,dt);self.p_now=-D_CHG_W

        if self.sim.service_requested and self.state in ("TAKEOFF","STRUCTURE"):
            self.wps=[];self.nav_goal=None;self.state="SERVICE_TO_BASE";self.service_msg="Returning to base for spare pack"
            self.sim.event("DRONE","battery service requested -> returning to BASE")
        if t-self.t_risk>=0.5:
            self.t_risk=t
            if airborne and t-self.t_dij>=2.0:
                self.t_dij=t;self.ret_dist=self.dist_home()      # real route length home on the LiDAR map
            self.e_ret=self.ret_energy(self.x,self.y,self.z) if airborne else 0.0
            self.pf=p_fail(self.bat,self.e_ret) if airborne else 0.0
            self.margin=self.bat.e-self.bat.reserve-self.e_ret
            if self.state in ("TAKEOFF","STRUCTURE") and self.pf>P_MAX and not self.sim.service_requested:
                self.state="RETURN";self.wps=[];self.nav_goal=None
                self.sim.event("DRONE",f"low battery risk -> BASE recharge (P={self.pf:.3f})")

        if t-self.t_hb>=0.5:
            self.t_hb=t;self.sim.send_tel(self.name)

        if airborne and self.z>0.35 and self.state=="STRUCTURE":
            self.camera_sample(t);self.record_structure(t)

        getattr(self,"st_"+self.state.lower())(dt,t)

    # ------------------------------------------------------------------ states
    def st_charge(self,dt,t):
        self.vx=self.vy=self.vz=0.0
        if self.sim.service_requested and self.bat.soc>=0.95:
            self.state="SERVICE_LOAD";self.service_msg="Spare battery loaded"
            return
        if self.bat.soc>=0.95:
            if self.todo_struct():
                self.state="TAKEOFF";self.sorties+=1
                self.sim.event("DRONE",f"take-off: structural inspection ({len(self.todo_struct())} zones)")
            else:self.state="DONE"

    def st_takeoff(self,dt,t):
        self.fly_to(self.x,self.y,ALT,dt)
        if self.z>ALT-0.05:self.state="STRUCTURE"

    def st_structure(self,dt,t):
        if self.struct_name is None:
            if not self.start_next_struct():
                self.state="RETURN";self.wps=[];self.nav_goal=None;return
        if self.sim.struct_done[self.struct_name]:
            self.struct_name=None;self.struct_points=[];self.wps=[];self.nav_goal=None;return
        if self.struct_point_i>=len(self.struct_points):
            self.sim.finish_struct_scan(self.struct_name)
            self.struct_name=None;self.struct_points=[];self.wps=[];self.nav_goal=None;return
        tx,ty=self.struct_points[self.struct_point_i]
        res=self.nav_to(tx,ty,ALT,dt,t)
        if res=="arrived":
            self.struct_point_i+=1
        elif res=="unreachable":                       # e.g. a rock-fall sealed this viewpoint: skip it
            self.struct_point_i+=1;self.skipped+=1

    def st_service_to_base(self,dt,t):
        if self._go_home(dt,t):
            self.vx=self.vy=self.vz=0.0
            self.state="CHARGE"          # charge to 95% first; st_charge then loads the spare pack
            self.service_msg="At BASE - charging before the battery delivery"

    def st_service_load(self,dt,t):
        self.vx=self.vy=self.vz=0.0
        if self.z>0.08:self.z=max(0.04,self.z-D_VZMAX*dt);return
        self.service_pack=SERVICE_PACK_WH
        self.service_msg=f"Spare pack loaded: {self.service_pack:.1f} Wh"
        self.state="SERVICE_TO_QUAD";self.wps=[];self.nav_goal=None
        self.sim.event("DRONE",f"spare pack loaded ({self.service_pack:.1f} Wh) -> flying to QUAD")

    def st_service_to_quad(self,dt,t):
        q=self.sim.quad
        if not q.alive:
            self.state="RETURN_SERVICE_BASE";self.sim.service_requested=False
            self.wps=[];self.nav_goal=None;return
        if self.z<ALT-0.05 and math.hypot(self.x-BASE_POS[0],self.y-BASE_POS[1])<0.5:
            self.fly_to(self.x,self.y,ALT,dt);return          # climb away from the pad first
        if math.hypot(q.x-self.x,q.y-self.y)>0.7:
            self.nav_to(q.x,q.y,ALT,dt,t,critical=True)       # dynamic route through the tunnels
            return
        self.nav_goal=None;self.wps=[]
        h=self.fly_to(q.x,q.y,SERVICE_DOCK_ALT,dt,vmax=min(D_VMAX,0.24),avoid=False)
        if h<0.15 and abs(self.z-SERVICE_DOCK_ALT)<0.12:
            self.state="SERVICE_TRANSFER";self.dwell=0.0

    def st_service_transfer(self,dt,t):
        q=self.sim.quad
        self.vx=self.vy=self.vz=0.0
        self.dwell+=dt
        if self.dwell<SERVICE_DWELL_S:return
        transfer=min(self.service_pack,SERVICE_TRANSFER_MAX_WH,max(0.0,q.bat.cap-q.bat.e))
        q.bat.e=min(q.bat.cap,q.bat.e+transfer)
        self.service_pack-=transfer
        self.service_msg=f"Transferred {transfer:.1f} Wh to QUAD"
        q.state="WAIT";q.wait_reason="battery delivered"
        self.sim.service_requested=False
        self.sim.service_count+=1
        self.sim.event("DRONE",f"battery delivery complete: +{transfer:.1f} Wh to QUAD; QUAD continues in place")
        self.state="RETURN_SERVICE_BASE";self.wps=[];self.nav_goal=None

    def st_return_service_base(self,dt,t):
        if self._go_home(dt,t):
            self.service_pack=0.0;self.vx=self.vy=self.vz=0.0
            self.state="CHARGE";self.service_msg="Service complete - recharging at base"
            self.sim.event("DRONE","returned to BASE after battery delivery - recharging")

    def st_return(self,dt,t):
        if self._go_home(dt,t):
            self.vx=self.vy=self.vz=0.0;self.state="CHARGE"
            self.sim.event("DRONE","landed at BASE - recharging")

    def st_done(self,dt,t):
        self.vx=self.vy=self.vz=0.0


# =====================================================================================
# 10.  VISUALISATION  (VPython)
# =====================================================================================
COL = {-1: color.gray(0.55), 0: vector(0.15, 0.75, 0.25), 1: vector(0.95, 0.8, 0.1), 2: vector(0.9, 0.15, 0.15)}
HAZ_COL = {"low_oxygen": vector(1.0, 0.62, 0.25), "high_temp": color.gray(0.55),
           "high_dust": color.gray(0.08), "high_gas": vector(0.45, 0.88, 0.45)}
HAZ_OUT = {"low_oxygen": color.orange, "high_temp": color.gray(0.3), "high_dust": color.black,
           "high_gas": vector(0.05, 0.5, 0.05)}
HAZ_TXT = {"low_oxygen": "LOW OXYGEN", "high_temp": "HIGH TEMP", "high_dust": "HIGH DUST", "high_gas": "HIGH GAS"}
OBS_COL = {"ROCK PILE": vector(0.42, 0.38, 0.34), "MACHINE": vector(0.88, 0.66, 0.05),
           "FALLEN TIMBER": vector(0.50, 0.32, 0.15), "ROCKFALL": vector(0.62, 0.22, 0.08)}


CHIP = {-1: "#8a8f98", 0: "#1fa64a", 1: "#d9a400", 2: "#d62828"}


# ---- uneven mine floor (VISUAL ONLY: it does not change planning, LiDAR or physics) -------
_TR = np.random.default_rng(ENV_SEED + 101)          # separate RNG: simulation random stream untouched
_MOUNDS = [(float(_TR.uniform(X0, X1)), float(_TR.uniform(Y0, Y1)), float(_TR.uniform(0.5, 1.7)),
            float(_TR.uniform(0.05, 0.17))) for _ in range(34)]


def terrain_h(x, y):
    """floor height [m] at plan position (x,y): ripples + muck/rubble mounds, flat at the base pad"""
    if ARGS.terrain <= 0:
        return 0.0
    h = (0.035 * math.sin(0.9 * x + 0.4) * math.cos(0.8 * y + 1.1) + 0.022 * math.sin(2.3 * x + 1.9 * y + 0.6)
         + 0.013 * math.sin(4.7 * x - 3.9 * y + 2.0) + 0.008 * math.sin(8.3 * x + 7.1 * y) + 0.075)
    for mx, my, r, a in _MOUNDS:
        dd = ((x - mx) ** 2 + (y - my) ** 2) / (r * r)
        if dd < 9.0:
            h += a * math.exp(-dd)
    f = min(1.0, max(0.0, (math.hypot(x - BASE_POS[0], y - BASE_POS[1]) - 0.8) / 1.2))
    return max(0.0, h * f) * ARGS.terrain


def terrain_col(x, y, h, top, shade=1.0):
    """earthy, varied floor colours: dusty tan zones, grey rock, dark wet muck spots"""
    t = min(1.0, h / max(top, 1e-6))
    patch = 0.5 + 0.5 * math.sin(0.55 * x + 1.3) * math.sin(0.47 * y + 0.2)
    damp = math.sin(1.7 * x - 0.9 * y + 2.1) * math.sin(1.1 * x + 1.9 * y)
    r, g, b = 0.48 + 0.26 * t + 0.20 * patch, 0.44 + 0.24 * t + 0.14 * patch, 0.38 + 0.20 * t + 0.03 * patch
    if damp > 0.50:
        k = (damp - 0.50) / 0.50
        r, g, b = r * (1 - 0.45 * k), g * (1 - 0.42 * k), b * (1 - 0.32 * k)
    return vector(r * shade, g * shade, b * shade)


TERR_TOP = max(terrain_h(X0 + i * 0.25, Y0 + j * 0.25) for i in range(int((X1 - X0) / 0.25) + 1)
               for j in range(int((Y1 - Y0) / 0.25) + 1)) + 0.01


def build_cracks():
    """roof cracks, placed where the structural model gives high crack severity (plan-view polylines)"""
    rr = np.random.default_rng(ENV_SEED + 303)
    out = []
    for name, (a, b, c, d) in STRUCT_ZONES.items():
        x = a + 0.3
        while x < b:
            y = c + 0.3
            while y < d:
                cr = structural_truth(name, x, y, 0.0)[0]
                if cr >= 0.40:
                    L = 0.35 + 1.1 * cr
                    ang = float(rr.uniform(0, 2 * math.pi))
                    px, py = x, y
                    pts = []
                    for _ in range(6):
                        pts.append((px, py))
                        ang += float(rr.normal(0, 0.55))
                        px += math.cos(ang) * L / 5
                        py += math.sin(ang) * L / 5
                    out.append(dict(zone=name, cr=cr, pts=pts, cx=x, cy=y))
                y += 0.7
            x += 0.7
    return out


CRACKS = build_cracks()


def V(x, y, h=0.0):
    """plan (x,y) + height  ->  VPython world (x, up, -y)"""
    return vector(x, h, -y)


def screen_size():
    """Best-effort screen size so the canvas fills the monitor (fallback 1600x900)."""
    try:
        import tkinter
        r = tkinter.Tk(); r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight(); r.destroy()
        return int(w), int(h)
    except Exception:
        return 1600, 900


def bat_text(name, o):
    if not o.alive:
        return f"{name:<6} DEAD"
    pct = o.bat.soc * 100
    n = int(round(pct / 10))
    bar = "█" * n + "░" * (10 - n)
    pw = o.p_now
    mode = "charging" if pw < 0 else "discharging"
    return (f"{name:<6} {bar} {pct:5.1f}%  {o.bat.voltage(max(pw, 0.1)):5.2f} V  "
            f"{abs(pw):5.1f} W {mode}")


class Viz:
    def __init__(self,sim):
        self.sim=sim
        sw,sh=screen_size()
        W=ARGS.width or max(1300,sw-20)
        H=ARGS.height or max(760,sh-80)
        self.W=W;self.H=H
        panel_w=ARGS.panel if ARGS.panel else max(600,min(760,int(W*0.40)))
        main_w=max(700,W-panel_w-34)
        main_h=max(560,H-245)
        self.scene=canvas(title="<b>Autonomous Mine Safety System - QUADRUPED LiDAR + STRUCTURAL DRONE</b>",
                          width=main_w,height=main_h,background=color.white,align="left")
        self.scene.center=V((X0+X1)/2,(Y0+Y1)/2,0)
        self.scene.forward=vector(0,-0.985,-0.17);self.scene.up=vector(0,1,0);self.scene.range=11.5

        box(canvas=self.scene,pos=V((X0+X1)/2,(Y0+Y1)/2,-0.01),
            size=vector(X1-X0,0.02,Y1-Y0),color=color.white)
        self._build_terrain(self.scene)
        self.map_cracks=[]
        for ck in CRACKS:
            self.map_cracks.append(curve(canvas=self.scene,pos=[V(px,py,TERR_TOP+0.04) for px,py in ck["pts"]],
                                         color=vector(0.95,0.05,0.05),radius=0.035+0.045*ck["cr"],visible=False))
        self.crack_seen=[False]*len(CRACKS)

        # IMPORTANT: hazard/structural regions are NOT drawn initially.
        # They are created only after the robot reaches the required coverage.
        self.hazard_objects={}
        self.struct_objects={}
        self.show_gt=True

        # Mine walls only.
        for a,b,c,d in WALLS:
            L=math.hypot(c-a,d-b)
            box(canvas=self.scene,pos=V((a+c)/2,(b+d)/2,VIS_WALL_H/2),
                axis=vector(c-a,0,-(d-b)),size=vector(L+0.08,VIS_WALL_H,0.09),
                color=color.gray(0.08))
        for ob in INIT_OBS:self._draw_obstacle(self.scene,ob,True)
        label(canvas=self.scene,pos=V(BASE_POS[0],BASE_POS[1],0.35),text="BASE / ENTRY",
              height=11,box=True,opacity=0.15,color=color.black)

        # Base marker.
        cylinder(canvas=self.scene,pos=V(BASE_POS[0],BASE_POS[1],0),axis=vector(0,0.05,0),
                 radius=0.24,color=vector(0.15,0.45,0.95))

        # QUADRUPED - deliberately larger and prominent.
        self.qbody=box(canvas=self.scene,pos=V(0,0,0.20),size=vector(0.72,0.18,0.34),color=color.blue)
        self.qhead=sphere(canvas=self.scene,pos=V(0,0,0.22),radius=0.15,color=color.cyan)
        self.qlid=cylinder(canvas=self.scene,pos=V(0,0,0.38),axis=vector(0,0.10,0),radius=0.055,color=color.red)
        self.qlegs=[cylinder(canvas=self.scene,pos=V(0,0,0.18),axis=vector(0,-0.20,0),
                             radius=0.022,color=color.black) for _ in range(4)]
        self.qtrail=curve(canvas=self.scene,color=color.blue,radius=0.014,retain=900)
        self.lidar=curve(canvas=self.scene,pos=[V(0,0,0.27)]*(2*100),
                         color=vector(0.1,0.55,1.0),radius=0.004)
        self.qtag=label(canvas=self.scene,pos=V(0,0,0.62),text="QUADRUPED",
                        height=13,box=True,opacity=0.15,color=color.black)

        # DRONE - large and visible.
        self.dbody=box(canvas=self.scene,pos=V(0,0,0.5),size=vector(0.38,0.12,0.38),color=color.orange)
        self.arms=[box(canvas=self.scene,pos=V(0,0,0.5),size=vector(0.72,0.025,0.035),color=color.gray(0.2)) for _ in range(2)]
        self.rot=[cylinder(canvas=self.scene,pos=V(0,0,0.5),axis=vector(0,0.012,0),
                           radius=0.14,color=color.gray(0.3),opacity=0.45) for _ in range(4)]
        self.shadow=cylinder(canvas=self.scene,pos=V(0,0,0.006),axis=vector(0,0.006,0),
                             radius=0.14,color=color.black,opacity=0.25)
        self.fov=curve(canvas=self.scene,pos=[V(0,0,0.04)]*4,color=color.orange,radius=0.006)
        self.dtrail=curve(canvas=self.scene,color=color.cyan,radius=0.012,retain=900)
        self.dtag=label(canvas=self.scene,pos=V(0,0,0.95),text="DRONE",
                        height=13,box=True,opacity=0.15,color=color.black)
        self.dlidar=curve(canvas=self.scene,pos=[V(0,0,0.5)]*(2*60),color=vector(1.0,0.55,0.0),radius=0.004)

        # LiDAR legend near the mine.
        self.lidar_note=label(canvas=self.scene,pos=V(1.8,0.6,0.08),
                              text="QUAD LiDAR 240-beam 5 m  |  DRONE LiDAR 180-beam 4 m",
                              height=9,box=True,opacity=0.08,color=color.blue)

        # ------------------------------------------------------------------
        # RIGHT COLUMN: BIG DRONE CAMERA + INFORMATION BOARD   |   GRAPHS under the 3D view
        # ------------------------------------------------------------------
        cw=panel_w-12
        self._build_camera(cw,int(cw*0.70))
        self._build_cam_world()
        self._build_info(cw,720)
        gw=max(300,(main_w-46)//3)
        self._build_graphs(gw,int(gw*0.66))

        self.msg=label(canvas=self.scene,pos=V(12,18,1.6),text="",height=18,box=True,opacity=0.0,color=color.red)
        self.tick=0
        self.scene.bind("keydown",self.on_key)

    def _build_terrain(self,cv,shade=1.0):
        """uneven floor: shaded height-field mesh + scattered rocks (static, drawn once per canvas)"""
        if ARGS.terrain<=0:return
        st=0.6;nx=int(round((X1-X0)/st));ny=int(round((Y1-Y0)/st));e=0.05
        top=max(TERR_TOP,0.05);vs={}
        for i in range(nx+1):
            for j in range(ny+1):
                x=X0+i*st;y=Y0+j*st;h=terrain_h(x,y)
                dhx=(terrain_h(x+e,y)-terrain_h(x-e,y))/(2*e);dhy=(terrain_h(x,y+e)-terrain_h(x,y-e))/(2*e)
                vs[(i,j)]=vertex(canvas=cv,pos=V(x,y,h),normal=vector(-3.5*dhx,1.0,3.5*dhy),
                                 color=terrain_col(x,y,h,top,shade))
        for i in range(nx):
            for j in range(ny):
                quad(canvas=cv,vs=[vs[(i,j)],vs[(i+1,j)],vs[(i+1,j+1)],vs[(i,j+1)]])
        rr=np.random.default_rng(ENV_SEED+202);n=0;tries=0
        while n<70 and tries<2000:
            tries+=1
            x=float(rr.uniform(X0+0.3,X1-0.3));y=float(rr.uniform(Y0+0.3,Y1-0.3))
            if math.hypot(x-BASE_POS[0],y-BASE_POS[1])<1.6 or not _free_xy(x,y):continue
            a=float(rr.uniform(0.05,0.15));g=float(rr.uniform(0.28,0.52))*shade
            sphere(canvas=cv,pos=V(x,y,terrain_h(x,y)+a*0.25),size=vector(a*2.0,a*1.1,a*1.6),
                   color=vector(g,g*0.95,g*0.85),axis=vector(math.cos(n),0,math.sin(n)))
            n+=1

    def _draw_obstacle(self,cv,ob,tag):
        """solid box (rock pile / machine / timber / rock-fall) standing on the uneven floor"""
        hgt=0.95 if tag else OBS_H
        col=OBS_COL.get(ob["kind"],color.gray(0.4))
        objs=[box(canvas=cv,pos=V(ob["cx"],ob["cy"],terrain_h(ob["cx"],ob["cy"])+hgt/2),
                  size=vector(ob["w"],hgt,ob["h"]),color=col)]
        if tag:
            objs.append(label(canvas=cv,pos=V(ob["cx"],ob["cy"],hgt+0.22),text=ob["kind"],height=9,
                              box=True,opacity=0.15,color=color.black))
        return objs

    def add_obstacle(self,ob):
        self._draw_obstacle(self.scene,ob,True)
        if getattr(self,"cam_world",False):self._draw_obstacle(self.panel,ob,False)

    def reveal_cracks(self,zone=None):
        """highlight cracks on the map: those the drone camera is looking at (or a whole zone once it is reported)"""
        d=self.sim.drone
        for i,ck in enumerate(CRACKS):
            if self.crack_seen[i]:continue
            if zone is not None:hit=(ck["zone"]==zone)
            else:hit=(d.alive and d.state=="STRUCTURE" and d.z>0.35 and ck["zone"]==d.struct_name
                      and math.hypot(d.x-ck["cx"],d.y-ck["cy"])<=1.3)
            if hit:
                self.crack_seen[i]=True;self.map_cracks[i].visible=True

    def on_key(self,ev):
        k=ev.key;s=self.sim
        if k=="k":s.kill("QUAD","manual kill")
        elif k=="d":s.kill("DRONE","manual kill")
        elif k in ("+","="):s.ts=min(20,s.ts*1.4)
        elif k in ("-","_"):s.ts=max(0.25,s.ts/1.4)
        elif k==" ":s.paused=not s.paused
        elif k in ("r","R"):s.manual_rockfall()

    def _haz_text(self,name):
        cls=self.sim.hazard_cls[name]
        if not self.sim.hazard_done[name]:return "NOT SCANNED"
        return f"{CLS_NAME[cls]} | {self.sim.hazard_coverage(name)*100:.0f}% | {self.sim.hazard_why[name]}"

    def _struct_text(self,name):
        if not self.sim.struct_done[name]:return "NOT SCANNED"
        return f"{CLS_NAME[self.sim.struct_cls[name]]} | {self.sim.struct_p[name]:.2f}"

    def update_hazard(self,name,cls,cov,why):
        if not self.sim.hazard_done[name]:return
        a,b,c,d=HAZ[name]
        col=COL[cls]
        objs=[
            box(canvas=self.scene,pos=V((a+b)/2,(c+d)/2,TERR_TOP+0.006),size=vector(b-a,0.012,d-c),
                color=col,opacity=0.55),
            curve(canvas=self.scene,pos=[V(a,c,TERR_TOP+0.025),V(b,c,TERR_TOP+0.025),V(b,d,TERR_TOP+0.025),V(a,d,TERR_TOP+0.025),V(a,c,TERR_TOP+0.025)],
                  color=col,radius=0.02),
            label(canvas=self.scene,pos=V((a+b)/2,(c+d)/2,TERR_TOP+0.08),
                  text=f"{HAZ_TXT[name]}\n{CLS_NAME[cls]}",height=11,box=True,opacity=0.15,color=color.black)
        ]
        if getattr(self,'cam_world',False):
            objs.append(box(canvas=self.panel,pos=V((a+b)/2,(c+d)/2,TERR_TOP+0.006),size=vector(b-a,0.012,d-c),color=col,opacity=0.55))
        self.hazard_objects[name]=objs

    def update_struct(self,name,cls,cov,crack,defl,pcol):
        a,b,c,d=STRUCT_ZONES[name]
        col=COL[cls]
        objs=[
            box(canvas=self.scene,pos=V((a+b)/2,(c+d)/2,TERR_TOP+0.008),size=vector(b-a,0.012,d-c),
                color=col,opacity=0.18),
            curve(canvas=self.scene,pos=[V(a,c,TERR_TOP+0.03),V(b,c,TERR_TOP+0.03),V(b,d,TERR_TOP+0.03),V(a,d,TERR_TOP+0.03),V(a,c,TERR_TOP+0.03)],
                  color=col,radius=0.015),
            label(canvas=self.scene,pos=V((a+b)/2,(c+d)/2,TERR_TOP+0.09),
                  text=f"{STRUCT_DISPLAY[name]}\n{CLS_NAME[cls]}  P={pcol:.2f}",
                  height=10,box=True,opacity=0.12,color=color.black)
        ]
        self.struct_objects[name]=objs
        if hasattr(self,'crack_seen'):self.reveal_cracks(name)

    # ------------------------------------------------------------------ info board
    def _build_info(self,cw,ch):
        p=self.info=canvas(title="<b>MISSION INFORMATION</b>",width=cw,height=ch,
                           background=vector(0.95,0.96,0.97),align="right")
        p.userzoom=False;p.userspin=False;p.userpan=False
        self.info_w,self.info_h=cw,ch
        self.ilab=[label(canvas=p,pos=vector(0,0,0),text="",xoffset=0,yoffset=0,height=12,align="left",
                         box=False,line=False,font="monospace",color=color.black,visible=False) for _ in range(110)]
        self.icache=[None]*len(self.ilab)

    @staticmethod
    def _bar(frac,n=10):
        k=int(round(max(0.0,min(1.0,frac))*n));return "█"*k+"░"*(n-k)

    def _info_rows(self,t):
        s=self.sim;q=s.quad;d=s.drone;r=s.radio
        HEAD=vector(0.05,0.20,0.55);GRY=vector(0.30,0.30,0.30);BLK=color.black
        TC={-1:vector(0.45,0.45,0.45),0:vector(0.05,0.55,0.15),1:vector(0.75,0.52,0.0),2:vector(0.85,0.10,0.10)}
        def soc_c(p):return TC[0] if p>50 else (TC[1] if p>25 else TC[2])
        rows=[]   # each row = (kind, [(text,colour,x_px),...])
        nh=sum(s.hazard_done.values());ns=sum(s.struct_done.values())
        rows.append(("n",[(f"t={t:7.1f}s  x{s.ts:.1f}{'  PAUSED' if s.paused else ''}",BLK,0),
                          (f"env {nh}/{len(HAZ)} {self._bar(nh/len(HAZ),6)}",BLK,230),
                          (f"struct {ns}/{len(STRUCT_ZONES)} {self._bar(ns/len(STRUCT_ZONES),6)}",BLK,460)]))
        rows.append(("h",[("ROBOTS",HEAD,0)]))
        for nm,o,ex in (("QUAD",q,f"LiDAR hits {q.lidar_hits}"),("DRONE",d,f"alt {d.z:.2f} m  sorties {d.sorties}")):
            if not o.alive:
                rows.append(("n",[(nm,BLK,0),("DEAD",TC[2],60)]));rows.append(("n",[]));continue
            pc=o.bat.soc*100;pw=o.p_now
            rows.append(("n",[(nm,BLK,0),(o.state,GRY,60),(f"{self._bar(o.bat.soc)} {pc:3.0f}%",soc_c(pc),190),
                              (f"{o.bat.voltage(max(pw,0.1)):.2f} V {abs(pw):4.1f} W{' chg' if pw<0 else ''}",BLK,400)]))
            rows.append(("n",[(f"dist {o.distance:6.1f} m   return-risk Pf {o.pf:.3f}   {ex}",GRY,60)]))
        rows.append(("h",[("LIVE SENSORS AT QUADRUPED",HEAD,0)]))
        try:g,o2,tp,du=[float(x) for x in s.ds.read(q.x,q.y,s.t)]
        except Exception:g=o2=tp=du=0.0
        def lv(v,c,u,low=False):
            bad=(v<u) if low else (v>u);warn=(v<c) if low else (v>c);return TC[2] if bad else (TC[1] if warn else TC[0])
        rows.append(("n",[(f"gas {g:5.1f} %LEL",lv(g,GAS_C,GAS_U),0),(f"O2 {o2:5.1f} %",lv(o2,O2_C,O2_U,True),175),
                          (f"temp {tp:5.1f} C",lv(tp,TMP_C,TMP_U),340),(f"dust {du:5.1f} mg/m3",lv(du,DST_C,DST_U),510)]))
        rows.append(("n",[(f"LiDAR  front {q.front_min:.2f}  left {q.left_clear:.2f}  right {q.right_clear:.2f} m",GRY,0)]))
        rows.append(("n",[(f"DRONE LiDAR front {d.front_min:.2f} m  nav {d.nav_status}  replans {d.replans}  stuck {d.stuck_total}",GRY,0)]))
        rows.append(("n",[(f"ENV seed {ENV_SEED}  obstacles {len(OBSTACLES)}  rock-falls pending {len(s.pending_falls)}  [R] = drop one now",GRY,0)]))
        rows.append(("h",[("ENVIRONMENT ZONES  (quadruped)",HEAD,0)]))
        for i,n in enumerate(("low_oxygen","high_temp","high_dust","high_gas"),1):
            if s.hazard_done[n]:
                rows.append(("n",[(f"{i}. {HAZ_TXT[n]}",BLK,0),(CLS_NAME[s.hazard_cls[n]],TC[s.hazard_cls[n]],200),(s.hazard_why[n][:34],GRY,300)]))
            else:
                cv=s.hazard_coverage(n)
                rows.append(("n",[(f"{i}. {HAZ_TXT[n]}",BLK,0),((f"SCANNING {cv*100:.0f}%" if cv>0 else "PENDING"),TC[-1],200)]))
        rows.append(("h",[("STRUCTURE ZONES  (drone camera)",HEAD,0)]))
        for i,n in enumerate(STRUCT_ORDER,1):
            if s.struct_done[n]:
                rows.append(("n",[(f"{i}. {STRUCT_DISPLAY[n]}",BLK,0),(CLS_NAME[s.struct_cls[n]],TC[s.struct_cls[n]],200),
                                  (f"P(collapse) {s.struct_p[n]:.2f}",GRY,300)]))
            else:
                cv=s.struct_coverage(n)
                rows.append(("n",[(f"{i}. {STRUCT_DISPLAY[n]}",BLK,0),((f"SCANNING {cv*100:.0f}%" if cv>0 else "PENDING"),TC[-1],200)]))
        rows.append(("h",[("DRONE CAMERA READING  (same values shown on the camera)",HEAD,0)]))
        zone=STRUCT_DISPLAY.get(d.struct_name,"TUNNEL TRANSIT");cc=TC[2] if d.camera_crack>=0.45 else TC[0]
        rows.append(("n",[(f"{zone} | {d.camera_status}" if d.alive else "DRONE OFFLINE",cc,0)]))
        rows.append(("n",[(f"crack {d.camera_crack:.2f}   deflection {d.camera_defl:.1f} mm   P(collapse) {d.camera_p:.2f}",cc,0)]))
        rows.append(("h",[("BASE KNOWLEDGE - 10 SECTORS",HEAD,0)]))
        for iy in range(NZY-1,-1,-1):
            rows.append(("n",[(f"S{iy*NZX+ix+1:<2d} {CLS_NAME[s.base_know.cls(iy*NZX+ix)]}",TC[s.base_know.cls(iy*NZX+ix)],ix*128) for ix in range(NZX)]))
        rows.append(("h",[("RADIO LINKS",HEAD,0)]))
        for a,b in (("QUAD","DRONE"),("BASE","QUAD"),("BASE","DRONE")):
            if s.alive(a) and s.alive(b):
                ps,via=r.route(a,b);ok=s.link_ok.get((a,b),True)
                rows.append(("n",[(f"{a}-{b}",BLK,0),(f"{r.rssi(a,b):6.1f} dBm  PDR {r.pdr(a,b):.2f}",GRY,150),
                                  ("UP" if ok else "DOWN",TC[0] if ok else TC[2],360),(f"via {via}" if via else "direct",GRY,420)]))
            else:rows.append(("n",[(f"{a}-{b}",BLK,0),("OFFLINE",TC[2],150)]))
        rows.append(("n",[(f"sent {r.sent}  delivered {r.delivered}  relayed {r.relayed}  retries {r.retries}  queue {len(r.pending)}",GRY,0)]))
        rows.append(("h",[("LATEST EVENTS",HEAD,0)]))
        for e in s.events[-5:][::-1]:rows.append(("n",[(e[:92],BLK,0)]))
        return rows

    def _update_panel(self,t):
        rows=self._info_rows(t)
        y=self.info_h/2-16;k=0;x0=-self.info_w/2+10
        for kind,cells in rows:
            if kind=="h":y-=7
            for text,col,xp in cells:
                if k>=len(self.ilab):break
                key=(text,y,xp,(col.x,col.y,col.z))
                lb=self.ilab[k]
                if self.icache[k]!=key:
                    lb.xoffset=x0+xp;lb.yoffset=y;lb.text=text;lb.color=col;lb.visible=True;self.icache[k]=key
                k+=1
            y-=17 if kind=="h" else 16
        while k<len(self.ilab):
            if self.icache[k] is not None:
                self.ilab[k].visible=False;self.icache[k]=None
            k+=1

    def draw(self,t):
        s=self.sim
        q=s.quad
        d=s.drone
        c,sn=math.cos(q.th),math.sin(q.th)
        moving=abs(q.v)>0.01
        hq=terrain_h(q.x,q.y)
        sl=(terrain_h(q.x+c*0.3,q.y+sn*0.3)-terrain_h(q.x-c*0.3,q.y-sn*0.3))/0.6
        self.qbody.pos=V(q.x,q.y,hq+(0.20+0.01*math.sin(2*q.phase) if moving else 0.20))
        self.qbody.axis=vector(c*0.40,sl*0.40,-sn*0.40)
        self.qhead.pos=V(q.x+c*0.22,q.y+sn*0.22,hq+0.22+sl*0.22)
        self.qlid.pos=V(q.x,q.y,hq+0.38)
        hips=[(0.16,0.10,0),(0.16,-0.10,math.pi),(-0.16,0.10,math.pi),(-0.16,-0.10,0)]
        for leg,(fx,ly,off) in zip(self.qlegs,hips):
            hx=q.x+c*fx-sn*ly;hy=q.y+sn*fx+c*ly
            sw=0.55*math.sin(q.phase+off) if moving else 0.0
            fxw=hx+c*0.20*math.sin(sw);fyw=hy+sn*0.20*math.sin(sw)
            leg.pos=V(hx,hy,hq+0.19);leg.axis=vector(fxw-hx,0.04+0.05*max(0,-math.cos(sw)),-(fyw-hy))
        self.qbody.color=color.red if not q.alive else color.blue
        self.qtag.pos=V(q.x,q.y+0.16,hq+0.62)
        self.qtag.text=f"QUADRUPED | {q.state} | {q.bat.soc*100:.0f}%"
        self.qtrail.append(V(q.x,q.y,hq+0.025))

        self.dbody.pos=V(d.x,d.y,d.z+0.03)
        self.dbody.axis=vector(math.cos(d.yaw)*0.20,0,-math.sin(d.yaw)*0.20)
        self.dbody.color=color.red if not d.alive else color.orange
        for k,arm in enumerate(self.arms):
            ang=d.yaw+math.pi/4+k*math.pi/2
            arm.pos=V(d.x,d.y,d.z+0.03);arm.axis=vector(math.cos(ang)*0.44,0,-math.sin(ang)*0.44)
        for k in range(4):
            ang=d.yaw+math.pi/4+k*math.pi/2
            rx=d.x+math.cos(ang)*0.23;ry=d.y+math.sin(ang)*0.23
            self.rot[k].pos=V(rx,ry,d.z+0.05)
        self.shadow.pos=V(d.x,d.y,terrain_h(d.x,d.y)+0.006)
        a1,a2=d.yaw-FRONT_FOV/2,d.yaw+FRONT_FOV/2
        for i,p in enumerate([V(d.x,d.y,0.04),
                              V(d.x+1.5*math.cos(a1),d.y+1.5*math.sin(a1),0.04),
                              V(d.x+1.5*math.cos(a2),d.y+1.5*math.sin(a2),0.04),
                              V(d.x,d.y,0.04)]):self.fov.modify(i,pos=p)
        self.fov.visible=d.z>0.3
        self.dtag.pos=V(d.x,d.y+0.16,d.z+0.45)
        self.dtag.text=f"DRONE | {d.state} | {d.bat.soc*100:.0f}%"
        self.dtrail.append(V(d.x,d.y,d.z))

        if q.last_rng is not None and self.tick%2==0:
            for i in range(100):
                k=i*2
                if k>=LIDAR_N:break
                a=q.ang[k]+q.th;r=q.last_rng[k]
                self.lidar.modify(2*i,pos=V(q.x,q.y,hq+0.27))
                self.lidar.modify(2*i+1,pos=V(q.x+r*math.cos(a),q.y+r*math.sin(a),hq+0.27))
        if self.tick%4==0:
            if d.alive and d.last_rng is not None and d.z>0.3:
                self.dlidar.visible=True
                for i in range(60):
                    k=i*3;a=d.last_wang[k];r=d.last_rng[k]
                    self.dlidar.modify(2*i,pos=V(d.x,d.y,d.z+0.03))
                    self.dlidar.modify(2*i+1,pos=V(d.x+r*math.cos(a),d.y+r*math.sin(a),d.z+0.03))
            else:self.dlidar.visible=False
        self.draw_camera(t)
        if self.tick%3==0:self.reveal_cracks()
        if self.tick%6==0:self._update_panel(t)
        self.tick+=1

    # ------------------------------------------------------------------ drone camera (real 3D first-person view)
    CAM_CEIL=1.6

    def _build_camera(self,cw,ch):
        p=self.panel=canvas(title="<b>DRONE ONBOARD CAMERA</b>",width=cw,height=ch,
                            background=vector(0.02,0.02,0.03),align="right")
        p.userzoom=False;p.userspin=False;p.userpan=False;p.autoscale=False
        p.fov=1.15;p.ambient=color.gray(0.55);p.up=vector(0,1,0)

    def _build_cam_world(self):
        """same mine as the main view (walls, uneven floor, rocks, hazard patches) + a dark roof with the real cracks"""
        p=self.panel;CH=self.CAM_CEIL
        box(canvas=p,pos=V((X0+X1)/2,(Y0+Y1)/2,CH),size=vector(X1-X0,0.04,Y1-Y0),color=vector(0.20,0.18,0.16))
        box(canvas=p,pos=V((X0+X1)/2,(Y0+Y1)/2,-0.02),size=vector(X1-X0,0.02,Y1-Y0),color=vector(0.35,0.32,0.28))
        for a,b,c,d in WALLS:
            L=math.hypot(c-a,d-b)
            box(canvas=p,pos=V((a+c)/2,(b+d)/2,CH/2),axis=vector(c-a,0,-(d-b)),
                size=vector(L+0.08,CH,0.09),color=vector(0.46,0.43,0.39))
        for ob in INIT_OBS:self._draw_obstacle(p,ob,False)
        self._build_terrain(p,0.9)
        cylinder(canvas=p,pos=V(BASE_POS[0],BASE_POS[1],0),axis=vector(0,0.05,0),radius=0.24,color=vector(0.15,0.45,0.95))
        # roof cracks: the same cracks that are highlighted on the map
        for ck in CRACKS:
            curve(canvas=p,pos=[V(px,py,CH-0.03) for px,py in ck["pts"]],color=vector(0.95,0.10,0.10),
                  radius=0.008+0.022*ck["cr"])
        self.cqbody=box(canvas=p,pos=V(0,0,0.2),size=vector(0.72,0.18,0.34),color=color.blue)
        self.cqhead=sphere(canvas=p,pos=V(0,0,0.22),radius=0.15,color=color.cyan)
        self.cam_world=True

    def draw_camera(self,t):
        s=self.sim;d=s.drone;q=s.quad;p=self.panel
        active=(d.alive and d.state=="STRUCTURE" and d.z>0.35)
        pitch=0.40 if active else -0.12              # looks up at the roof while inspecting
        cp=math.cos(pitch)
        fw=vector(math.cos(d.yaw)*cp,math.sin(pitch),-math.sin(d.yaw)*cp)
        pos=V(d.x,d.y,d.z+terrain_h(d.x,d.y)+0.04)
        try:
            p.camera.pos=pos;p.camera.axis=fw*4
        except Exception:
            p.center=pos+fw*4;p.forward=fw
        c,sn=math.cos(q.th),math.sin(q.th);hq=terrain_h(q.x,q.y)
        self.cqbody.pos=V(q.x,q.y,hq+0.20);self.cqbody.axis=vector(c*0.40,0,-sn*0.40)
        self.cqhead.pos=V(q.x+c*0.22,q.y+sn*0.22,hq+0.22)
        self.cqbody.color=color.red if not q.alive else color.blue
        if self.tick%6==0:
            zone=STRUCT_DISPLAY.get(d.struct_name,"TUNNEL TRANSIT");cr=d.camera_crack
            if not d.alive:
                p.title="<b>DRONE ONBOARD CAMERA</b> | <span style='color:#d62828'>DRONE OFFLINE - NO SIGNAL</span>"
            else:
                flag=" | <b style='color:#d62828'>CRACK DETECTED</b>" if (active and cr>=0.45) else ""
                p.title=(f"<b>DRONE ONBOARD CAMERA</b> | {zone} | {d.camera_status}{flag}<br>"
                         f"crack <b>{cr:.2f}</b> | deflection <b>{d.camera_defl:.1f} mm</b> | P(collapse) <b>{d.camera_p:.2f}</b>"
                         f" &nbsp;|&nbsp; ALT {d.z:.2f} m | SPD {math.hypot(d.vx,d.vy):.2f} m/s | "
                         f"YAW {math.degrees(d.yaw)%360:.0f}&deg; | BAT {d.bat.soc*100:.0f}%")

    # ------------------------------------------------------------------ live graphs
    def _build_graphs(self,gw,gh):
        self.G={};self.gcount=0
        BL,OR,GR,RD,PU,TE=(vector(0.1,0.35,0.95),vector(0.95,0.5,0.05),vector(0.1,0.65,0.25),
                           vector(0.85,0.12,0.12),vector(0.55,0.2,0.75),vector(0.0,0.6,0.65))
        def mk(key,title,ytitle,series,ymin=None,ymax=None):
            kw=dict(width=gw,height=gh,title="<b>"+title+"</b>",xtitle="time [s]",ytitle=ytitle,
                    align="left",fast=True,background=color.white,foreground=color.black)
            if ymin is not None:kw["ymin"]=ymin
            if ymax is not None:kw["ymax"]=ymax
            g=graph(**kw)
            self.G[key]=[(k,gcurve(graph=g,color=col,label=lab)) for k,col,lab in series]
        mk("soc","Battery state of charge","SOC [%]",[("q_soc",BL,"QUAD"),("d_soc",OR,"DRONE")],0,100)
        mk("pow","Power draw (negative = charging)","P [W]",[("q_p",BL,"QUAD"),("d_p",OR,"DRONE")])
        mk("gas","Quad sensors: gas and dust","%LEL / mg/m3",[("q_gas",RD,"gas %LEL"),("q_dust",PU,"dust")])
        mk("tmp","Quad sensors: temperature and O2","degC / %",[("q_temp",OR,"temp degC"),("q_o2",TE,"O2 %")])
        mk("str","Drone camera: structural risk","0 - 1",[("d_crack",RD,"crack severity"),("d_pcol",PU,"P(collapse)")],0,1)
        mk("rssi","Radio signal strength","RSSI [dBm]",[("rssi_qd",GR,"QUAD-DRONE"),("rssi_bq",BL,"BASE-QUAD"),("rssi_bd",OR,"BASE-DRONE")])
        mk("pdr","Radio delivery probability","PDR",[("pdr_qd",GR,"QUAD-DRONE"),("pdr_bq",BL,"BASE-QUAD"),("pdr_bd",OR,"BASE-DRONE")],0,1.05)
        mk("pf","Return-to-base failure risk","Pf",[("q_pf",BL,"QUAD"),("d_pf",OR,"DRONE")],0)
        mk("zones","Sector knowledge at BASE","sectors",[("known",color.black,"known"),("safe",GR,"safe"),
                                                           ("caution",vector(0.85,0.65,0.0),"caution"),("unsafe",RD,"unsafe")],0,10)

    def graphs(self,t,nk,ns,nu):
        self.gcount+=1
        if self.gcount%2:return
        row=self.sim.tele[-1]
        for curves in self.G.values():
            for k,gc in curves:
                try:v=float(row[k])
                except Exception:continue
                if math.isfinite(v):gc.plot(t,max(-130.0,v))

    def final(self,text):
        self.msg.text=text



# =====================================================================================
# 11.  SIMULATION
# =====================================================================================
class Sim:
    def __init__(self):
        self.t=0.0;self.ts=TS;self.paused=False
        self.ds=Dataset(ARGS.dataset);self.truth=self.ds.zone_truth()
        self.radio=Radio(self)
        self.quad=Quad(self);self.drone=Drone(self)
        self.base_know=Knowledge()
        self.last_seen={};self.dead=set();self.events=[];self.link_ok={}
        self.tele=[];self.disc=[];self.reports=[];self.evlog=[]
        self.service_requested=False;self.service_count=0
        self.pending_falls=sorted(PENDING_FALLS,key=lambda o:o["t_on"])
        self.hazard_tries={name:0 for name in HAZ};self.struct_tries={name:0 for name in STRUCT_ZONES}

        self.hazard_done={name:False for name in HAZ}
        self.hazard_visited={name:set() for name in HAZ}
        self.hazard_sum={name:np.zeros(4) for name in HAZ}
        self.hazard_n={name:0 for name in HAZ}
        self.hazard_cls={name:-1 for name in HAZ}
        self.hazard_why={name:"" for name in HAZ}

        self.struct_done={name:False for name in STRUCT_ZONES}
        self.struct_visited={name:set() for name in STRUCT_ZONES}
        self.struct_sum={name:np.zeros(3) for name in STRUCT_ZONES}
        self.struct_n={name:0 for name in STRUCT_ZONES}
        self.struct_cls={name:-1 for name in STRUCT_ZONES}
        self.struct_p={name:0.0 for name in STRUCT_ZONES}
        self.struct_last={name:(0.0,0.0,0.0) for name in STRUCT_ZONES}

        self.viz=Viz(self)
        for _ob in INIT_OBS:self.sync_obstacle_effects(_ob)       # v97: cells squeezed against a starting obstacle count as covered
        self.t_log=self.t_rad=self.t_mon=0.0
        self.event("SYS","simulation started - QUAD owns environmental scan; DRONE owns structural camera scan")
        self.event("ENV",f"seed {ENV_SEED}: {len(INIT_OBS)} obstacles on the floor, {len(self.pending_falls)} rock-falls scheduled; "
                         f"drone map prior = {ARGS.drone_prior}")
        for ob in INIT_OBS:self.sync_obstacle_effects(ob)

    def pos3(self,n):
        if n=="BASE":return (BASE_POS[0],BASE_POS[1],0.1)
        if n=="QUAD":return (self.quad.x,self.quad.y,0.2)
        return (self.drone.x,self.drone.y,self.drone.z)

    def alive(self,n):
        return n=="BASE" or n not in self.dead

    def kill(self,n,why):
        if n in self.dead:return
        self.dead.add(n);o=self.quad if n=="QUAD" else self.drone
        o.alive=False;o.state="DEAD";self.event(n,f"*** DEAD ({why}) at ({o.x:.2f},{o.y:.2f})")

    def event(self,who,text):
        s=f"[{self.t:6.1f}s] {who}: {text}"
        self.events.append(s);self.evlog.append((self.t,who,text));print(s)

    def request_battery_service(self):
        if self.service_requested:return
        self.service_requested=True
        self.drone.service_msg="QUAD requested spare battery"
        if self.drone.state=="DONE":self.drone.state="CHARGE"      # at the pad, charged: st_charge loads the pack

    def send_zone(self,node,src,z,cls,vals,n,cells=(),why=""):
        other="DRONE" if node=="QUAD" else "QUAD"
        msg={"kind":"ZONE","src":src,"zone":z,"cls":cls,"vals":vals,"n":n,"t":self.t,"from":node,
             "cells":sorted(cells),"why":why}
        self.reports.append((self.t,node,z,cls,n))
        self.radio.send(node,msg,["BASE",other])
        (self.quad if node=="QUAD" else self.drone).know.merge(msg)

    def send_hazard(self,name,cls,mean,coverage,why):
        msg={"kind":"HAZARD","name":name,"cls":cls,"mean":mean,"coverage":coverage,"why":why,"t":self.t,"from":"QUAD"}
        self.radio.send("QUAD",msg,["BASE","DRONE"])
        self.hazard_cls[name]=cls;self.hazard_why[name]=why
        self.viz.update_hazard(name,cls,coverage,why)

    def send_struct(self,name,cls,crack,defl,pcol,coverage):
        msg={"kind":"STRUCT","name":name,"cls":cls,"crack":crack,"defl":defl,"pcol":pcol,"coverage":coverage,"t":self.t,"from":"DRONE"}
        self.radio.send("DRONE",msg,["BASE","QUAD"])
        self.struct_cls[name]=cls;self.struct_p[name]=pcol
        self.viz.update_struct(name,cls,coverage,crack,defl,pcol)

    def record_hazard_sample(self,x,y,v):
        for name,rect in HAZ.items():
            a,b,c,d=rect
            if not (a<=x<=b and c<=y<=d):continue
            pts=HAZ_SCAN_POINTS[name]
            if not pts:continue
            # Sensor footprint is about 0.45 m; mark all coarse cells touched.
            touched=[i for i,p in enumerate(pts) if math.dist((x,y),p)<=0.48]
            if touched:
                self.hazard_visited[name].update(touched)
                self.hazard_sum[name]+=np.asarray(v,float)
                self.hazard_n[name]+=1

    def hazard_coverage(self,name):
        n=max(1,len(HAZ_SCAN_POINTS[name]))
        return len(self.hazard_visited[name])/n

    def finish_hazard_scan(self,name):
        if self.hazard_done[name]:return
        cov=self.hazard_coverage(name);self.hazard_tries[name]+=1
        k=self.hazard_tries[name]
        if cov<(0.90 if k<=3 else 0.75 if k<=6 else 0.40):     # v96: never loop forever on a sealed-off cell
            self.event("QUAD",f"{name.upper()} coverage {cov*100:.0f}% - continue scanning")
            return
        mean=self.hazard_sum[name]/max(1,self.hazard_n[name])
        cls=classify(*mean)
        vals=np.asarray(mean)
        lv=var_levels(vals)
        idx=int(np.argmax(lv))
        why=f"{VAR_NAMES[idx]} ({vals[idx]:.1f} {VAR_UNITS[idx]})" if lv[idx]>0 else "all readings normal"
        self.hazard_done[name]=True
        self.send_hazard(name,cls,mean.round(2).tolist(),cov,why)
        self.event("QUAD",f"{name.replace('_',' ').upper()} FULLY SCANNED {cov*100:.0f}% -> {CLS_NAME[cls]} ({why})")

    def all_hazards_done(self):
        return all(self.hazard_done.values())

    def record_struct_sample(self,name,x,y,t):
        if name not in STRUCT_ZONES:return
        pts=STRUCT_PTS[name]
        touched=[]
        radius=max(0.70,self.drone.footprint*0.45)
        for i,p in enumerate(pts):
            if math.dist((x,y),p)<=radius:touched.append(i)
        if touched:self.struct_visited[name].update(touched)
        truth=np.asarray(structural_truth(name,x,y,t),float)
        self.struct_sum[name]+=truth;self.struct_n[name]+=1

    def struct_coverage(self,name):
        pts=STRUCT_PTS[name]
        return len(self.struct_visited[name])/max(1,len(pts))

    def finish_struct_scan(self,name):
        if self.struct_done[name]:return
        cov=self.struct_coverage(name);self.struct_tries[name]+=1
        k=self.struct_tries[name]
        if cov<(STRUCT_SCAN_FRAC if k<=3 else 0.60 if k<=6 else 0.35):
            return
        mean=self.struct_sum[name]/max(1,self.struct_n[name])
        crack,defl,pcol=map(float,mean)
        cls=structural_class(pcol)
        self.struct_done[name]=True;self.struct_cls[name]=cls;self.struct_p[name]=pcol;self.struct_last[name]=(crack,defl,pcol)
        self.send_struct(name,cls,crack,defl,pcol,cov)
        self.event("DRONE",f"{STRUCT_DISPLAY[name]} FULLY SCANNED {cov*100:.0f}% -> {CLS_NAME[cls]} P(collapse)={pcol:.2f}")

    def send_tel(self,node):
        o=self.quad if node=="QUAD" else self.drone
        self.radio.send(node,{"kind":"TEL","t":self.t,"x":o.x,"y":o.y,"soc":o.bat.soc},["BASE"],ttl=6.0)

    def deliver(self,dest,m,src,hop2):
        if m["kind"]=="TEL":
            self.last_seen[src]=(self.t,m["x"],m["y"],m["soc"]);return
        if m["kind"]=="HAZARD":
            if dest=="BASE":
                self.hazard_cls[m["name"]]=m["cls"];self.hazard_why[m["name"]]=m["why"]
                self.viz.update_hazard(m["name"],m["cls"],m["coverage"],m["why"])
            return
        if m["kind"]=="STRUCT":
            if dest=="BASE":
                self.struct_cls[m["name"]]=m["cls"];self.struct_p[m["name"]]=m["pcol"]
                self.viz.update_struct(m["name"],m["cls"],m["coverage"],m["crack"],m["defl"],m["pcol"])
            return
        if dest=="BASE":
            old=self.base_know.cls(m["zone"]);self.base_know.merge(m);new=self.base_know.cls(m["zone"])
            if new!=old:
                self.disc.append((self.t,m["zone"],new,m["src"],self.t-m["t"]))
                self.event("BASE",f"zone {m['zone']+1} -> {CLS_NAME[new]} (from {m['src']}, latency {self.t-m['t']:.1f}s)")
        elif dest=="QUAD":self.quad.know.merge(m)
        elif dest=="DRONE":self.drone.know.merge(m)

    # ---------------------------------------------------------------- dynamic environment
    def sync_obstacle_effects(self,ob):
        """scan / viewpoint cells that sit under or right next to an obstacle can no longer be visited: count them as covered"""
        a,b,c,d=ob["rect"];m=0.60                  # v97: 0.6 m = body radius + inflation -> unreachable
        for name in HAZ:
            for i,p in enumerate(HAZ_SCAN_POINTS[name]):
                if a-m<=p[0]<=b+m and c-m<=p[1]<=d+m:self.hazard_visited[name].add(i)
        for name in STRUCT_ZONES:
            for i,p in enumerate(STRUCT_PTS[name]):
                if a-m<=p[0]<=b+m and c-m<=p[1]<=d+m:self.struct_visited[name].add(i)

    def drop_obstacle(self,ob,why="ROCKFALL"):
        activate_obstacle(ob);self.sync_obstacle_effects(ob)
        self.viz.add_obstacle(ob)
        self.event("ENV",f"{why} at ({ob['cx']:.1f},{ob['cy']:.1f}) size {ob['w']:.1f} x {ob['h']:.1f} m - robots must re-plan")

    def spawn_rockfalls(self):
        if not self.pending_falls or self.t<self.pending_falls[0]["t_on"]:return
        ob=self.pending_falls[0];need=max(ob["w"],ob["h"])/2+1.2
        for o in (self.quad,self.drone):                      # never drop it on top of a robot: wait 10 s
            if o.alive and math.hypot(o.x-ob["cx"],o.y-ob["cy"])<need:
                ob["t_on"]=self.t+10.0;self.pending_falls.sort(key=lambda p:p["t_on"]);return
        self.pending_falls.pop(0);self.drop_obstacle(ob)

    def manual_rockfall(self):
        """key R: a rock-fall 2-4 m from the robots (always leaves every zone reachable)"""
        occ=TRUE_OCC.copy();rects=[o["rect"] for o in OBSTACLES]
        for p in self.pending_falls:occ|=_rect_mask(p["rect"]);rects.append(p["rect"])
        for o in (self.quad,self.drone,self.quad):
            if not o.alive:continue
            for _ in range(12):
                ob=_candidate(ENV_RNG,occ,rects,near=(o.x,o.y))
                if ob is None:break
                need=max(ob["w"],ob["h"])/2+1.2
                if all((not r.alive) or math.hypot(r.x-ob["cx"],r.y-ob["cy"])>=need for r in (self.quad,self.drone)):
                    ob["kind"]="ROCKFALL";self.drop_obstacle(ob,"ROCKFALL (manual)");return
        self.event("ENV","no safe place found for a rock-fall right now")

    def monitor_links(self):
        for a,b in (("QUAD","DRONE"),("BASE","QUAD"),("BASE","DRONE")):
            if not(self.alive(a) and self.alive(b)):continue
            p,via=self.radio.route(a,b);ok=self.link_ok.get((a,b),True)
            if ok and p<LINK_DOWN:
                self.link_ok[(a,b)]=False;self.event("RADIO",f"{a}<->{b} LINK DOWN - store/forward")
            elif not ok and p>LINK_UP:
                self.link_ok[(a,b)]=True;self.event("RADIO",f"{a}<->{b} LINK RESTORED")
            else:self.link_ok[(a,b)]=ok

    def step(self,dt):
        self.t+=dt
        self.spawn_rockfalls()
        self.quad.step(dt,self.t);self.drone.step(dt,self.t)
        if self.t-self.t_rad>=RADIO_TICK:self.t_rad=self.t;self.radio.step()
        if self.t-self.t_mon>=1.0:self.t_mon=self.t;self.monitor_links()
        if self.t-self.t_log>=1.0:self.t_log=self.t;self.log()

    def log(self):
        q,d,r=self.quad,self.drone,self.radio;kn=[self.base_know.cls(z) for z in range(NZ)]
        env_now=self.ds.read(q.x,q.y,self.t)
        self.tele.append(dict(
            t=self.t,q_x=q.x,q_y=q.y,q_soc=q.bat.soc*100,q_state=q.state,q_eret=q.e_ret*1000,
            q_gas=float(env_now[0]),q_o2=float(env_now[1]),q_temp=float(env_now[2]),q_dust=float(env_now[3]),
            d_crack=float(d.camera_crack),d_defl=float(d.camera_defl),d_pcol=float(d.camera_p),
            q_pf=q.pf,q_margin=q.margin*1000,q_p=q.p_now,q_v=q.v,q_volt=q.bat.voltage(max(q.p_now,0.1)),
            q_front=q.front_min,q_lidar_hits=q.lidar_hits,
            d_x=d.x,d_y=d.y,d_z=d.z,d_soc=d.bat.soc*100,d_state=d.state,d_eret=d.e_ret*1000,
            d_pf=d.pf,d_margin=d.margin*1000,d_p=d.p_now,d_volt=d.bat.voltage(max(d.p_now,0.1)),
            rssi_qd=r.rssi("QUAD","DRONE"),rssi_bq=r.rssi("BASE","QUAD"),rssi_bd=r.rssi("BASE","DRONE"),
            pdr_qd=r.pdr("QUAD","DRONE"),pdr_bq=r.pdr("BASE","QUAD"),pdr_bd=r.pdr("BASE","DRONE"),
            queue=len(r.pending),known=sum(1 for k in kn if k>=0),safe=kn.count(0),caution=kn.count(1),unsafe=kn.count(2),
            haz_done=sum(self.hazard_done.values()),struct_done=sum(self.struct_done.values()),
            service_count=self.service_count,d_front=d.front_min,d_replans=d.replans,d_stuck=d.stuck_total,
            d_hits=d.lidar_hits,d_collisions=d.collisions,n_obst=len(OBSTACLES)))
        if hasattr(self.viz,"graphs"):self.viz.graphs(self.t,self.tele[-1]["known"],kn.count(0),kn.count(2))

    def finished(self):
        return self.quad.state in ("DONE","DEAD") and self.drone.state in ("DONE","DEAD")

    def run(self):
        sub=0.0
        while self.t<T_MAX and not self.finished():
            rate(FPS)
            if self.paused:continue
            sub+=self.ts/FPS
            while sub>=DT:self.step(DT);sub-=DT
            self.viz.draw(self.t)
        self.log();self.viz.draw(self.t);self.summary()
        self.viz.final("MISSION COMPLETE - outputs saved in "+ARGS.outdir)
        make_graphs(self)

    def summary(self):
        print("\n================ MINE ROBOT SUMMARY ================")
        print(f"time {self.t:.0f}s | environmental scans {sum(self.hazard_done.values())}/{len(HAZ)} | "
              f"structural scans {sum(self.struct_done.values())}/{len(STRUCT_ZONES)}")
        print(f"QUAD distance {self.quad.distance:.1f} m | SOC {self.quad.bat.soc*100:.1f}% | "
              f"LiDAR hits {self.quad.lidar_hits} | battery services {self.service_count}")
        print(f"DRONE distance {self.drone.distance:.1f} m | SOC {self.drone.bat.soc*100:.1f}% | "
              f"state {self.drone.state}")
        d=self.drone
        print(f"ENV seed {ENV_SEED} | obstacles {len(OBSTACLES)} (rock-falls pending {len(self.pending_falls)}) | drone prior {ARGS.drone_prior}")
        print(f"DRONE navigation: replans {d.replans} | stuck events {d.stuck_total} | viewpoints skipped {d.skipped} | contact-guard hits {d.collisions}")
        print(f"radio sent {self.radio.sent} delivered {self.radio.delivered} relayed {self.radio.relayed} retries {self.radio.retries}")


# =====================================================================================
# 12.  GRAPHS
# =====================================================================================
def make_graphs(sim):
    import matplotlib
    if not ARGS.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out = ARGS.outdir
    os.makedirs(out, exist_ok=True)
    tele = sim.tele
    if len(tele) < 3:
        return
    A = lambda k: np.array([r[k] for r in tele])
    t = A("t")
    ext = [X0, X1, Y0, Y1]
    clsmap = ListedColormap(["#26bf40", "#f2cc19", "#e62626"])

    def walls(ax):
        for (a, b, c, d) in WALLS:
            ax.plot([a, c], [b, d], "k-", lw=2)
        for ob in OBSTACLES:
            a_, b_, c_, d_ = ob["rect"]
            ax.fill([a_, b_, b_, a_], [c_, c_, d_, d_], color="#b5651d", alpha=0.9, zorder=3)
        ax.set_aspect("equal"); ax.set_xlim(X0 - 0.1, X1 + 0.1); ax.set_ylim(Y0 - 0.1, Y1 + 0.1)

    with open(os.path.join(out, "telemetry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tele[0].keys())); w.writeheader(); w.writerows(tele)
    with open(os.path.join(out, "zone_reports.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s", "node", "zone", "class", "n_samples"])
        for r in sim.reports: w.writerow([round(r[0], 2), r[1], r[2] + 1, CLS_NAME[r[3]], r[4]])

    ds = sim.ds
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    mid = ds.T // 2
    free = ~TRUE_OCC
    for a, (k, ttl, cm) in zip(ax.flat[:4], [(0, "Gas [%LEL]", "inferno"), (1, "O2 [%]", "viridis"),
                                              (2, "Temperature [C]", "hot"), (3, "Dust [mg/m3]", "copper")]):
        im = a.imshow(np.where(free, ds.v[mid, :, :, k], np.nan), origin="lower", extent=ext, cmap=cm)
        plt.colorbar(im, ax=a, fraction=0.046); a.set_title(ttl + f"  (t={mid*DS_SLICE_S:.0f}s)"); walls(a)
    ax.flat[4].imshow(np.where(free, ds.label[mid], np.nan), origin="lower", extent=ext, cmap=clsmap, vmin=0, vmax=2)
    ax.flat[4].set_title("Dataset label per cell (green safe / yellow caution / red unsafe)"); walls(ax.flat[4])
    zt = np.array(sim.truth).reshape(NZY, NZX)
    ax.flat[5].imshow(zt, origin="lower", extent=[X0, X1, Y0, Y1], cmap=clsmap, vmin=0, vmax=2)
    for z in range(NZ):
        cx, cy = zone_center(z); ax.flat[5].text(cx, cy, f"Z{z+1}", ha="center", va="center", fontsize=9)
    ax.flat[5].set_title("25 zones - truth derived from dataset"); walls(ax.flat[5])
    fig.tight_layout(); fig.savefig(os.path.join(out, "1_dataset_overview.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    lab = ds.raw[:, 10].astype(int); fr = ds.raw[:, 11] == 1
    for a, (col, ttl, thr) in zip(ax.flat, [(6, "Gas [%LEL]", (GAS_C, GAS_U)), (7, "O2 [%]", (O2_C, O2_U)),
                                           (8, "Temperature [C]", (TMP_C, TMP_U)), (9, "Dust [mg/m3]", (DST_C, DST_U))]):
        for l, cn, cc in [(0, "safe", "#26bf40"), (1, "caution", "#f2cc19"), (2, "unsafe", "#e62626")]:
            m = fr & (lab == l)
            if m.any(): a.hist(ds.raw[m, col], bins=40, alpha=0.7, color=cc, label=f"{cn} (n={m.sum()})")
        for th in thr: a.axvline(th, color="k", ls="--", lw=1)
        a.set_title(ttl + " by dataset label"); a.set_yscale("log"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "2_dataset_distributions.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    ax[0].plot(t, A("q_soc"), "b", label="Quadruped"); ax[0].plot(t, A("d_soc"), color="orange", label="Drone")
    ax[0].axhline(Q_RESERVE * 100, color="b", ls=":"); ax[0].axhline(D_RESERVE * 100, color="orange", ls=":")
    ax[0].set_ylabel("SOC [%]"); ax[0].legend(); ax[0].set_title("Battery state of charge (dotted = reserve)")
    for (te, who, tx) in sim.evlog:
        if "RETURN" in tx or "DEAD" in tx or "docked" in tx or "landed" in tx:
            ax[0].axvline(te, color="b" if who == "QUAD" else "orange", alpha=0.25)
    ax[1].plot(t, A("q_p"), "b", label="Quadruped power"); ax[1].plot(t, A("d_p"), color="orange", label="Drone power")
    ax[1].set_ylabel("Power [W] (negative = charging)"); ax[1].legend(); ax[1].set_title("Power draw")
    ax[2].plot(t, A("q_eret"), "b", label="E needed to return (quad)"); ax[2].plot(t, A("d_eret"), color="orange",
                                                                                   label="E needed to return (drone)")
    ax[2].plot(t, A("q_margin"), "b--", label="margin quad (usable - needed)")
    ax[2].plot(t, A("d_margin"), "--", color="orange", label="margin drone")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_ylabel("mWh"); ax[2].legend(ncol=2)
    ax[2].set_title("Return-to-base energy budget (operational range test)")
    ax[3].semilogy(t, np.maximum(A("q_pf"), 1e-6), "b", label="quad"); ax[3].semilogy(t, np.maximum(A("d_pf"), 1e-6),
                                                                                      color="orange", label="drone")
    ax[3].axhline(P_MAX, color="r", ls="--", label=f"P_MAX = {P_MAX}"); ax[3].set_ylabel("P(depletion before dock)")
    ax[3].set_xlabel("t [s]"); ax[3].legend(); ax[3].set_title("Risk of battery depletion on the way home")
    fig.tight_layout(); fig.savefig(os.path.join(out, "3_battery_and_return.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].stackplot(t, A("safe"), A("caution"), A("unsafe"), colors=["#26bf40", "#f2cc19", "#e62626"],
                    labels=["safe", "caution", "unsafe"])
    ax[0].set_title("Zones known at BASE while the robots are still moving"); ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("zones"); ax[0].legend(loc="upper left"); ax[0].set_ylim(0, NZ)
    d_t = sorted(r[0] for r in sim.reports if r[1] == "DRONE"); q_t = sorted(r[0] for r in sim.reports if r[1] == "QUAD")
    ax[1].step(d_t, np.arange(1, len(d_t) + 1), color="orange", label="drone reports")
    ax[1].step(q_t, np.arange(1, len(q_t) + 1), color="b", label="quadruped reports")
    if sim.disc:
        lat = [r[4] for r in sim.disc]
        ax[1].set_title(f"Reports streamed (mean latency to base {np.mean(lat):.1f}s)")
    ax[1].set_xlabel("t [s]"); ax[1].legend()
    kn = np.array([sim.base_know.cls(z) for z in range(NZ)]).reshape(NZY, NZX).astype(float)
    kn[kn < 0] = np.nan
    ax[2].imshow(kn, origin="lower", extent=ext, cmap=clsmap, vmin=0, vmax=2)
    for z in range(NZ):
        cx, cy = zone_center(z)
        k = sim.base_know.cls(z)
        ax[2].text(cx, cy, f"Z{z+1}\n{CLS_NAME[k][:4]}\n(truth {CLS_NAME[sim.truth[z]][:4]})", ha="center", va="center", fontsize=7,
                   color="k" if k == sim.truth[z] else "magenta")
    walls(ax[2]); ax[2].set_title("Final map at base (magenta = differs from dataset truth)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "4_mission_zone_discovery.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].plot(t, A("rssi_qd"), label="drone <-> quad"); ax[0].plot(t, A("rssi_bq"), label="base <-> quad")
    ax[0].plot(t, A("rssi_bd"), label="base <-> drone"); ax[0].axhline(SENS_DBM, color="r", ls="--", label="sensitivity")
    ax[0].set_ylabel("RSSI [dBm]"); ax[0].legend(ncol=4); ax[0].set_title("Radio links (path loss + wall attenuation)")
    ax[1].plot(t, A("pdr_qd"), label="drone <-> quad"); ax[1].plot(t, A("pdr_bq"), label="base <-> quad")
    ax[1].plot(t, A("pdr_bd"), label="base <-> drone")
    ax[1].set_ylabel("direct packet delivery ratio"); ax[1].legend(ncol=3)
    ax[2].plot(t, A("queue"), "k"); ax[2].set_ylabel("store-and-forward queue"); ax[2].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(os.path.join(out, "5_communication.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    g = sim.quad.grid.L
    ax[0].imshow(np.where(g < -0.3, 1.0, np.where(g > 0.8, 0.0, 0.5)), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    ax[0].plot(A("q_x"), A("q_y"), "b-", lw=1.5, label="quadruped path"); ax[0].legend()
    ax[0].set_title("Quadruped LiDAR occupancy map (white free / black wall / grey unknown)")
    walls(ax[1]); sc = ax[1].scatter(A("d_x"), A("d_y"), c=A("d_z"), s=6, cmap="plasma"); plt.colorbar(sc, ax=ax[1], label="drone height [m]")
    ax[1].plot(A("q_x"), A("q_y"), "b-", lw=1.2, label="quadruped"); ax[1].plot(*BASE_POS, "g^", ms=12, label="base")
    ax[1].legend(); ax[1].set_title("Drone flight (colour = height) and quadruped path")
    fig.tight_layout(); fig.savefig(os.path.join(out, "6_trajectories_lidar_map.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (src, ttl) in zip(ax, (("drone", "Drone (stand-off sensing)"), ("ground", "Quadruped (in-situ)"))):
        M = np.zeros((3, 3), int)
        for z, c in (sim.base_know.src[src]).items():
            M[sim.truth[z], c] += 1
        a.imshow(M, cmap="Blues")
        for i in range(3):
            for j in range(3): a.text(j, i, M[i, j], ha="center", va="center", fontsize=14)
        a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(["safe", "caution", "unsafe"])
        a.set_yticklabels(["safe", "caution", "unsafe"]); a.set_xlabel("reported"); a.set_ylabel("dataset truth"); a.set_title(ttl)
    fig.tight_layout(); fig.savefig(os.path.join(out, "7_confusion_drone_vs_ground.png"), dpi=110); plt.close(fig)
    # Final environmental hazard map: ONLY results from completed quadruped scans are colored.
    fig, ax = plt.subplots(figsize=(12, 8))
    walls(ax)
    status_colors={0:"#2ca02c",1:"#f1c40f",2:"#d62728"}
    for i,(name,rect) in enumerate(HAZ.items(),1):
        a,b,c,d=rect
        if sim.hazard_done[name]:
            cls=sim.hazard_cls[name]; face=status_colors[cls]; alpha=0.55
            ax.add_patch(plt.Rectangle((a,c),b-a,d-c,facecolor=face,edgecolor=face,alpha=alpha,lw=2))
            label=f"ZONE {i}\n{HAZ_TXT[name]}\n{CLS_NAME[cls]}"
        else:
            ax.add_patch(plt.Rectangle((a,c),b-a,d-c,facecolor="none",edgecolor="0.55",hatch="///",lw=1.5))
            label=f"ZONE {i}\nNOT COMPLETED"
        ax.text((a+b)/2,(c+d)/2,label,ha="center",va="center",fontsize=10,fontweight="bold")
    ax.plot(A("q_x"),A("q_y"),"b-",lw=1.6,label="quadruped path")
    ax.plot(A("d_x"),A("d_y"),color="orange",lw=1.2,label="drone tunnel path")
    ax.plot(*BASE_POS,"k^",ms=10,label="base")
    ax.set_title("FINAL ENVIRONMENTAL HAZARD MAP — QUADRUPED SCAN RESULTS")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out,"8_final_environment_map.png"),dpi=150); plt.close(fig)

    # Final structural map: only completed drone inspections receive a classification color.
    fig, ax = plt.subplots(figsize=(12,8)); walls(ax)
    for i,(name,rect) in enumerate(STRUCT_ZONES.items(),1):
        a,b,c,d=rect
        if sim.struct_done[name]:
            cls=sim.struct_cls[name]; face=status_colors[cls]
            ax.add_patch(plt.Rectangle((a,c),b-a,d-c,facecolor=face,edgecolor=face,alpha=0.45,lw=2))
            label=f"STRUCT ZONE {i}\n{STRUCT_DISPLAY[name]}\n{CLS_NAME[cls]}"
        else:
            ax.add_patch(plt.Rectangle((a,c),b-a,d-c,facecolor="none",edgecolor="0.55",hatch="///",lw=1.5))
            label=f"STRUCT ZONE {i}\nNOT COMPLETED"
        ax.text((a+b)/2,(c+d)/2,label,ha="center",va="center",fontsize=9,fontweight="bold")
    ax.plot(A("d_x"),A("d_y"),color="orange",lw=1.6,label="drone tunnel/inspection path")
    ax.plot(A("q_x"),A("q_y"),"b-",lw=1.0,label="quadruped path")
    ax.plot(*BASE_POS,"k^",ms=10,label="base")
    ax.set_title("FINAL STRUCTURAL INSPECTION MAP — DRONE CAMERA RESULTS")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out,"9_final_structural_map.png"),dpi=150); plt.close(fig)

    # Mission progress graph.
    fig, ax = plt.subplots(figsize=(12,6))
    ht=np.array([r["haz_done"] for r in tele]); st=np.array([r["struct_done"] for r in tele])
    ax.step(t,ht,where="post",label="environmental zones completed")
    ax.step(t,st,where="post",label="structural zones completed")
    ax.set_ylim(0,max(4,int(max(ht.max(),st.max()))+1)); ax.set_xlabel("Time [s]"); ax.set_ylabel("Completed zones")
    ax.set_title("Inspection Progress"); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out,"10_inspection_progress.png"),dpi=150); plt.close(fig)

    # -------------------------------------------------------------------------
    # Additional presentation-quality engineering graphs
    # -------------------------------------------------------------------------
    # 11) Environmental sensor trends at the quadruped
    fig, ax = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    sensor_specs = [
        ("q_gas", "Gas [%LEL]", (GAS_C, GAS_U), "Gas concentration measured along the quadruped route"),
        ("q_o2", "Oxygen [%]", (O2_C, O2_U), "Oxygen level measured along the quadruped route"),
        ("q_temp", "Temperature [°C]", (TMP_C, TMP_U), "Temperature measured along the quadruped route"),
        ("q_dust", "Dust [mg/m³]", (DST_C, DST_U), "Dust concentration measured along the quadruped route"),
    ]
    for a,(key,ylabel,thr,title) in zip(ax.flat,sensor_specs):
        a.plot(t,A(key),lw=1.5,label="quadruped sensor")
        a.axhline(thr[0],ls="--",lw=1,label="caution threshold")
        a.axhline(thr[1],ls=":",lw=1,label="unsafe threshold")
        a.set_ylabel(ylabel); a.set_title(title); a.grid(alpha=0.25)
        a.legend(fontsize=8)
    ax[1,0].set_xlabel("Time [s]"); ax[1,1].set_xlabel("Time [s]")
    fig.suptitle("Environmental Sensor Measurements During Mine Inspection",fontsize=15,fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(out,"11_environmental_sensor_trends.png"),dpi=150); plt.close(fig)

    # 12) Robot motion performance: speed and cumulative distance
    fig, ax = plt.subplots(2,1,figsize=(13,8),sharex=True)
    ax[0].plot(t,A("q_v"),lw=1.5,label="quadruped speed")
    ax[0].plot(t,np.gradient(A("d_x"))**2*0 + np.hypot(np.gradient(A("d_x"),t),np.gradient(A("d_y"),t)),lw=1.2,label="drone ground speed")
    ax[0].set_ylabel("Speed [m/s]"); ax[0].set_title("Robot Motion Performance"); ax[0].grid(alpha=0.25); ax[0].legend()
    qdist=np.r_[0,np.cumsum(np.hypot(np.diff(A("q_x")),np.diff(A("q_y"))))]
    ddist=np.r_[0,np.cumsum(np.hypot(np.diff(A("d_x")),np.diff(A("d_y"))))]
    ax[1].plot(t,qdist,lw=1.6,label="quadruped cumulative distance")
    ax[1].plot(t,ddist,lw=1.6,label="drone cumulative ground distance")
    ax[1].set_ylabel("Distance [m]"); ax[1].set_xlabel("Time [s]"); ax[1].grid(alpha=0.25); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out,"12_robot_motion_performance.png"),dpi=150); plt.close(fig)

    # 13) LiDAR clearance / obstacle sensitivity
    fig, ax = plt.subplots(figsize=(13,6))
    ax.plot(t,A("q_front"),lw=1.6,label="front clearance")
    ax.axhline(LIDAR_STOP_DIST,ls="--",lw=1.2,label="stop distance")
    ax.axhline(LIDAR_SLOW_DIST,ls=":",lw=1.2,label="slow distance")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Distance [m]")
    ax.set_title("LiDAR Obstacle Clearance and Local Navigation Response")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out,"13_lidar_clearance.png"),dpi=150); plt.close(fig)

    # 14) Structural inspection risk from the drone camera model
    fig, ax = plt.subplots(2,1,figsize=(13,8),sharex=True)
    ax[0].plot(t,A("d_crack"),lw=1.5,label="crack severity")
    ax[0].axhline(0.45,ls="--",lw=1,label="camera detection level")
    ax[0].set_ylabel("Crack severity [0–1]"); ax[0].set_title("Drone Structural Inspection")
    ax[0].grid(alpha=0.25); ax[0].legend()
    ax[1].plot(t,A("d_pcol"),lw=1.6,label="estimated collapse probability")
    ax[1].axhline(STRUCT_CAUTION,ls="--",lw=1,label="caution")
    ax[1].axhline(STRUCT_DANGER,ls=":",lw=1,label="unsafe")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("P(collapse)")
    ax[1].set_ylim(0,1); ax[1].grid(alpha=0.25); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out,"14_structural_risk.png"),dpi=150); plt.close(fig)

    # Three presentation-ready mission graphs.
    sensor = np.array([sim.ds.read(r["q_x"], r["q_y"], r["t"]) for r in tele])
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(t, sensor[:,0], label="Gas [%LEL]", lw=1.8)
    ax.plot(t, sensor[:,1], label="Oxygen [%]", lw=1.8)
    ax.plot(t, sensor[:,2], label="Temperature [C]", lw=1.8)
    ax.plot(t, sensor[:,3], label="Dust [mg/m3]", lw=1.8)
    ax.set_title("Environmental Sensor Readings During Inspection"); ax.set_xlabel("Mission time [s]"); ax.set_ylabel("Sensor value"); ax.grid(alpha=0.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out,"11_environmental_sensor_readings.png"),dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(t, A("q_front"), label="Front LiDAR clearance [m]", lw=1.8)
    ax.axhline(LIDAR_SLOW_DIST, ls="--", lw=1.2, label="Slow-down distance")
    ax.axhline(LIDAR_STOP_DIST, ls="--", lw=1.2, label="Stop distance")
    ax.set_title("Quadruped LiDAR Obstacle Clearance"); ax.set_xlabel("Mission time [s]"); ax.set_ylabel("Distance [m]"); ax.set_ylim(bottom=0); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out,"12_lidar_obstacle_clearance.png"),dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(t, A("q_soc"), label="Quadruped battery SOC [%]", lw=2)
    ax.plot(t, A("d_soc"), label="Drone battery SOC [%]", lw=2)
    ax.plot(t, np.array([r["haz_done"] for r in tele])/max(1,len(HAZ))*100, label="Environmental scan progress [%]", lw=2)
    ax.plot(t, np.array([r["struct_done"] for r in tele])/max(1,len(STRUCT_ZONES))*100, label="Structural scan progress [%]", lw=2)
    ax.set_title("Mission Progress and Battery Status"); ax.set_xlabel("Mission time [s]"); ax.set_ylabel("Percentage [%]"); ax.set_ylim(0,105); ax.grid(alpha=0.25); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out,"13_mission_progress_battery.png"),dpi=150); plt.close(fig)

    # -------------------------------------------------------------------------
    # Simple presentation graphs — styled like the reference screenshot.
    # -------------------------------------------------------------------------
    # 15) Zones known at BASE (streamed)
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.plot(t, A("known"), color="0.35", lw=2.2, label="known")
    ax.plot(t, A("safe"), color="#00d9e6", lw=2.2, label="safe")
    ax.plot(t, A("unsafe"), color="red", lw=2.2, label="unsafe")
    ax.set_title("Zones known at BASE (streamed)", fontsize=15)
    ax.set_xlabel("t [s]"); ax.set_ylabel("zones")
    ax.set_ylim(0, NZ + 1); ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "15_zones_known_at_base.png"), dpi=150, facecolor="white"); plt.close(fig)

    # 16) Battery state of charge
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.plot(t, A("q_soc"), color="blue", lw=2.2, label="Quadruped")
    ax.plot(t, A("d_soc"), color="orange", lw=2.2, label="Drone")
    ax.axhline(Q_RESERVE * 100, color="blue", ls="--", lw=1.1, alpha=0.65, label="Quadruped reserve")
    ax.axhline(D_RESERVE * 100, color="orange", ls="--", lw=1.1, alpha=0.65, label="Drone reserve")
    ax.set_title("Battery state of charge", fontsize=15)
    ax.set_xlabel("t [s]"); ax.set_ylabel("SOC [%]")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "16_battery_state_of_charge.png"), dpi=150, facecolor="white"); plt.close(fig)

    # 17) Inspection progress
    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    ax.plot(t, A("haz_done"), color="#00d9e6", lw=2.2, label="Environmental zones completed")
    ax.plot(t, A("struct_done"), color="red", lw=2.2, label="Structural zones completed")
    ax.set_title("Inspection progress", fontsize=15)
    ax.set_xlabel("t [s]"); ax.set_ylabel("zones completed")
    ax.set_ylim(0, max(len(HAZ), len(STRUCT_ZONES)) + 1)
    ax.grid(True, alpha=0.28); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "17_inspection_progress_simple.png"), dpi=150, facecolor="white"); plt.close(fig)

    # 18) drone LiDAR map + path (dynamic navigation)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    g = sim.drone.grid.L
    ax[0].imshow(np.where(g < -0.3, 1.0, np.where(g > 0.8, 0.0, 0.5)), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    ax[0].plot(A("d_x"), A("d_y"), color="orange", lw=1.2, label="drone path"); ax[0].legend()
    ax[0].set_title("Drone LiDAR occupancy map (white free / black wall+obstacle / grey unknown)")
    walls(ax[1]); ax[1].plot(A("d_x"), A("d_y"), color="orange", lw=1.2, label="drone path")
    ax[1].plot(A("q_x"), A("q_y"), "b-", lw=1.0, label="quadruped path"); ax[1].plot(*BASE_POS, "g^", ms=12, label="base")
    ax[1].legend(); ax[1].set_title(f"Dynamic environment (brown = obstacles / rock-falls), layout seed {ENV_SEED}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "18_drone_lidar_map_dynamic.png"), dpi=130); plt.close(fig)

    # 19) drone obstacle clearance + re-planning
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(t, A("d_front"), lw=1.4, label="drone front clearance")
    ax[0].axhline(D_CLEAR_SLOW, ls=":", label="slow-down"); ax[0].axhline(D_CLEAR_STOP, ls="--", label="stop")
    for (te, who, tx) in sim.evlog:
        if who == "ENV": ax[0].axvline(te, color="brown", alpha=0.5)
    ax[0].set_ylabel("m"); ax[0].set_title("Drone LiDAR clearance (brown lines = rock-falls)"); ax[0].legend(); ax[0].grid(alpha=0.25)
    ax[1].plot(t, A("d_replans"), label="replans (cumulative)"); ax[1].plot(t, A("d_stuck"), label="stuck events")
    ax[1].set_xlabel("t [s]"); ax[1].legend(); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(out, "19_drone_dynamic_navigation.png"), dpi=130); plt.close(fig)

    print(f"[graphs] saved final maps + graphs + telemetry.csv + zone_reports.csv in ./{out}")
    if ARGS.show:
        plt.show()


# =====================================================================================
if __name__ == "__main__":
    sim = Sim()
    sim.run()
    if not os.environ.get("MINESIM_NOWAIT"):
        while True:
            rate(5)