"""Forensic-style handcrafted features: strokes, loops, dots, slant, parts.

Everything works on the height-normalised binary mask (ink = 255).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize

N_ORIENT_BINS = 18


@dataclass
class Features:
    aspect: float            # width / height of the ink bounding box
    density: float           # ink pixels / bbox pixels
    skeleton_len: float      # skeleton length / bbox diagonal
    stroke_w: float          # mean stroke width / height
    slant_deg: float         # dominant stroke slant, + = leaning right
    orient_hist: np.ndarray  # stroke-direction histogram (18 bins over 0..180 deg)
    n_loops: int
    loop_areas: np.ndarray   # top-3 loop areas / bbox area (padded with 0)
    n_dots: int
    dots: np.ndarray         # (n, 2) normalised centroids
    n_endpoints: int
    n_junctions: int
    grid: np.ndarray         # 4x8 ink-density layout
    parts: np.ndarray        # (n, 8) connected-component descriptors, sorted left->right
    n_underlines: int = 0        # long near-horizontal strokes in the lower half
    underline_len: float = 0.0   # longest underline width / ink width
    underline_y: float = 0.0     # its vertical position (0 = top, 1 = bottom)


def _detect_underline(m: np.ndarray):
    """Long, thin, roughly horizontal stroke in the lower half of the signature."""
    h, w = m.shape
    L = max(15, int(0.35 * w))
    opened = np.zeros_like(m)
    for slope in (-0.17, -0.08, 0.0, 0.08, 0.17):          # flat or up/down-hill (about +-10 deg)
        dy = int(round(slope * L))
        kh = abs(dy) + 3
        k = np.zeros((kh, L), np.uint8)
        y0 = kh // 2 - dy // 2
        cv2.line(k, (0, y0), (L - 1, y0 + dy), 1, 1)
        opened = np.maximum(opened, cv2.morphologyEx(m, cv2.MORPH_OPEN, k))
    n, _, stats, cent = cv2.connectedComponentsWithStats(opened, connectivity=8)
    found = []
    for i in range(1, n):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if cw >= 0.35 * w and ch <= 0.2 * h and cent[i][1] >= 0.5 * h:
            found.append((cw / w, cent[i][1] / h))
    if not found:
        return 0, 0.0, 0.0
    longest = max(found)
    return len(found), float(longest[0]), float(longest[1])


def extract_features(mask: np.ndarray) -> Features:
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    area_total = float(h * w)

    skel = skeletonize(m > 0).astype(np.uint8)
    skel_len = float(skel.sum()) / float(np.hypot(h, w))
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    stroke_w = float(2.0 * dt[skel > 0].mean() / h) if skel.any() else 0.0

    # ---- stroke orientation & slant -------------------------------------------------
    f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    phi = (np.arctan2(gy, gx) + np.pi / 2.0) % np.pi        # stroke direction, 0..pi
    sel = mag > 0.05 * max(float(mag.max()), 1e-6)
    hist, _ = np.histogram(phi[sel], bins=N_ORIENT_BINS, range=(0, np.pi), weights=mag[sel])
    hist = hist / max(hist.sum(), 1e-9)
    near_vertical = sel & (phi > np.pi / 6) & (phi < 5 * np.pi / 6)
    if near_vertical.any():
        wgt = mag[near_vertical]
        slant = float(np.degrees((wgt * phi[near_vertical]).sum() / wgt.sum()) - 90.0)
    else:
        slant = 0.0

    # ---- loops (holes inside the ink) -----------------------------------------------
    min_loop = 0.0002 * area_total
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    loops = []
    if hier is not None:
        for c, hh in zip(cnts, hier[0]):
            if hh[3] != -1:
                a = cv2.contourArea(c)
                if a >= min_loop:
                    loops.append(a / area_total)
    loops = sorted(loops, reverse=True)
    loop_areas = np.zeros(3, np.float32)
    loop_areas[:min(3, len(loops))] = loops[:3]

    # ---- connected components: dots + letter-like parts ---------------------------
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    dots, parts = [], []
    for rank, i in enumerate(comps):
        x, y, cw, ch, a = stats[i]
        if rank > 0 and 6 <= a <= 0.004 * area_total:
            ar, fill = cw / max(ch, 1), a / max(cw * ch, 1)
            if 0.4 <= ar <= 2.5 and fill >= 0.5:
                dots.append([cent[i][0] / w, cent[i][1] / h])
        if rank < 40 and a >= 0.0003 * area_total:
            sub = (labels[y:y + ch, x:x + cw] == i).astype(np.uint8)
            hu = cv2.HuMoments(cv2.moments(sub, binaryImage=True)).flatten()[:3]
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12) / 10.0
            parts.append([cent[i][0] / w, cent[i][1] / h, cw / w, ch / h, a / area_total, *hu])
    parts = np.array(sorted(parts, key=lambda p: p[0]), np.float32).reshape(-1, 8)

    # ---- skeleton topology -----------------------------------------------------------
    nb = cv2.filter2D(skel.astype(np.float32), -1, np.ones((3, 3), np.float32),
                      borderType=cv2.BORDER_CONSTANT) - skel
    ends = (skel == 1) & (nb == 1)
    junc = ((skel == 1) & (nb >= 3)).astype(np.uint8)
    n_junc = cv2.connectedComponents(cv2.dilate(junc, np.ones((3, 3), np.uint8)))[0] - 1

    grid = cv2.resize(m.astype(np.float32), (8, 4), interpolation=cv2.INTER_AREA).flatten()

    n_ul, ul_len, ul_y = _detect_underline(m)

    return Features(
        aspect=w / h, density=float(m.sum()) / area_total, skeleton_len=skel_len,
        stroke_w=stroke_w, slant_deg=slant, orient_hist=hist.astype(np.float32),
        n_loops=len(loops), loop_areas=loop_areas,
        n_dots=len(dots), dots=np.array(dots, np.float32).reshape(-1, 2),
        n_endpoints=int(ends.sum()), n_junctions=int(max(n_junc, 0)),
        grid=grid, parts=parts,
        n_underlines=n_ul, underline_len=ul_len, underline_y=ul_y,
    )


# ======================================================================================
#  Comparison
# ======================================================================================
def _exp_sim(diff: float, scale: float) -> float:
    return float(np.exp(-abs(diff) / scale))


def _rel_sim(a: float, b: float, scale: float = 0.35) -> float:
    d = abs(a - b) / max(abs(a), abs(b), 1e-9)
    return float(np.exp(-d / scale))


def stroke_overlay(mask_a: np.ndarray, mask_b: np.ndarray, size=(128, 256), tau: float = 10.0):
    """Symmetric chamfer distance between the two skeletons on a common canvas."""
    def skel(m):
        s = cv2.resize(m, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return skeletonize(s > 127).astype(np.uint8)

    sa, sb = skel(mask_a), skel(mask_b)
    if sa.sum() == 0 or sb.sum() == 0:
        return 0.0, 99.0
    dt_b = cv2.distanceTransform((1 - sb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_a = cv2.distanceTransform((1 - sa).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (dt_b[sa > 0].mean() + dt_a[sb > 0].mean())
    return float(np.exp(-chamfer / tau)), float(chamfer)


def _parts_score(pa: np.ndarray, pb: np.ndarray) -> float:
    if len(pa) == 0 and len(pb) == 0:
        return 1.0
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    wts = np.array([1.5, 1.0, 1.0, 1.0, 0.5, 0.3, 0.3, 0.3], np.float32)
    cost = np.linalg.norm((pa * wts)[:, None, :] - (pb * wts)[None, :, :], axis=2)
    cost = np.minimum(cost, 1.0)
    r, c = linear_sum_assignment(cost)
    total = cost[r, c].sum() + abs(len(pa) - len(pb)) * 1.0   # unmatched part = full penalty
    return float(max(0.0, 1.0 - total / max(len(pa), len(pb))))


def _dots_score(fa: Features, fb: Features) -> float:
    if fa.n_dots == 0 and fb.n_dots == 0:
        return 1.0
    count = _exp_sim(fa.n_dots - fb.n_dots, 1.0)
    if fa.n_dots == 0 or fb.n_dots == 0:
        return 0.5 * count
    cost = np.linalg.norm(fa.dots[:, None, :] - fb.dots[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return 0.5 * count + 0.5 * _exp_sim(float(cost[r, c].mean()), 0.15)


def _loops_score(fa: Features, fb: Features) -> float:
    count = _exp_sim(fa.n_loops - fb.n_loops, 1.5)
    denom = float(fa.loop_areas.sum() + fb.loop_areas.sum())
    area = 1.0 if denom < 1e-9 else 1.0 - float(np.abs(fa.loop_areas - fb.loop_areas).sum()) / denom
    return 0.5 * count + 0.5 * max(area, 0.0)


def _underline_score(fa: Features, fb: Features) -> float:
    if fa.n_underlines == 0 and fb.n_underlines == 0:
        return 1.0
    if fa.n_underlines == 0 or fb.n_underlines == 0:
        return 0.0
    return (0.4 * _exp_sim(fa.n_underlines - fb.n_underlines, 1.0)
            + 0.3 * _exp_sim(fa.underline_len - fb.underline_len, 0.25)
            + 0.3 * _exp_sim(fa.underline_y - fb.underline_y, 0.15))


def compare_features(fa: Features, fb: Features, mask_a: np.ndarray, mask_b: np.ndarray):
    """Returns (scores 0..1 per feature, raw details, human-readable reasons)."""
    overlay, chamfer = stroke_overlay(mask_a, mask_b)
    slant_diff = abs(fa.slant_deg - fb.slant_deg)
    corr = np.corrcoef(fa.grid, fb.grid)[0, 1] if fa.grid.std() > 0 and fb.grid.std() > 0 else 0.0

    scores = {
        "stroke_overlay": overlay,
        "parts": _parts_score(fa.parts, fb.parts),
        "loops": _loops_score(fa, fb),
        "orientation": float(np.minimum(fa.orient_hist, fb.orient_hist).sum()),
        "slant": _exp_sim(slant_diff, 12.0),
        "dots": _dots_score(fa, fb),
        "underline": _underline_score(fa, fb),
        "aspect": _exp_sim(np.log(fa.aspect) - np.log(fb.aspect), 0.25),
        "stroke_width": _rel_sim(fa.stroke_w, fb.stroke_w),
        "endpoints": _rel_sim(fa.n_endpoints, fb.n_endpoints),
        "junctions": _rel_sim(fa.n_junctions, fb.n_junctions),
        "ink_layout": float(max(corr, 0.0)),
    }
    details = {
        "chamfer_px": round(chamfer, 2),
        "slant_deg": [round(fa.slant_deg, 1), round(fb.slant_deg, 1)],
        "loops": [fa.n_loops, fb.n_loops],
        "dots": [fa.n_dots, fb.n_dots],
        "underlines": [fa.n_underlines, fb.n_underlines],
        "endpoints": [fa.n_endpoints, fb.n_endpoints],
        "junctions": [fa.n_junctions, fb.n_junctions],
        "parts": [len(fa.parts), len(fb.parts)],
        "aspect": [round(fa.aspect, 2), round(fb.aspect, 2)],
        "stroke_width": [round(fa.stroke_w, 4), round(fb.stroke_w, 4)],
    }
    reasons = []
    if scores["slant"] < 0.5:
        reasons.append(f"Slant differs by {slant_diff:.0f} degrees")
    if fa.n_loops != fb.n_loops:
        reasons.append(f"Loop count differs ({fa.n_loops} vs {fb.n_loops})")
    if fa.n_dots != fb.n_dots:
        reasons.append(f"Dot count differs ({fa.n_dots} vs {fb.n_dots})")
    if (fa.n_underlines == 0) != (fb.n_underlines == 0):
        reasons.append("Underline present in only one of the signatures")
    elif fa.n_underlines and scores["underline"] < 0.5:
        reasons.append("Underline differs in length, position or count")
    if scores["parts"] < 0.5:
        reasons.append("Letter-like parts do not line up in shape/position")
    if scores["stroke_overlay"] < 0.4:
        reasons.append(f"Stroke paths deviate (mean {chamfer:.1f}px on a 256px-wide canvas)")
    if scores["aspect"] < 0.5:
        reasons.append("Overall width/height proportion differs")
    if scores["stroke_width"] < 0.5:
        reasons.append("Stroke thickness differs (different pen / pressure)")
    return scores, details, reasons
