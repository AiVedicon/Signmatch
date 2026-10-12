"""Runs without Django/torch: checks preprocessing, features and scoring behave sensibly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from signatures.compare import compare
from signatures.preprocess import prepare


def draw_signature(seed=0, slant=0.0, jitter=0.0, thickness=3, loops=2, dot=True, noise=0.0, underline=False):
    rng = np.random.default_rng(seed)
    img = np.full((300, 700, 3), 235, np.uint8)
    pts = []
    for k in range(loops):                       # loops built from a small circle path
        cx = 120 + k * 140
        for t in np.linspace(0, 2 * np.pi, 40):
            pts.append((cx + 45 * np.cos(t) + t * 12, 150 - 60 * np.sin(t)))
    pts += [(120 + loops * 140 + i * 15, 150 + 30 * np.sin(i / 2.0)) for i in range(15)]
    pts = np.array(pts, np.float32)
    pts[:, 0] += (150 - pts[:, 1]) * slant       # shear = slant
    pts += rng.normal(0, jitter, pts.shape)
    cv2.polylines(img, [pts.astype(np.int32)], False, (20, 20, 20), thickness, cv2.LINE_AA)
    if dot:
        cv2.circle(img, (160, 60), 7, (20, 20, 20), -1)
    if underline:
        cv2.line(img, (100, 262), (620, 250), (20, 20, 20), thickness, cv2.LINE_AA)
    if noise:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def score(a, b):
    return compare(prepare(a), prepare(b))


genuine = draw_signature(0)
same_noisy = draw_signature(0, jitter=0.6, noise=8)      # same signature, scan noise
variation = draw_signature(0, jitter=1.5, slant=0.03)    # natural variation
forgery = draw_signature(0, slant=0.45, loops=1, dot=False, thickness=6)  # wrong slant/loops/dot/pen

r_same = score(genuine, same_noisy)
r_var = score(genuine, variation)
r_forge = score(genuine, forgery)
for name, r in [("same+noise", r_same), ("natural variation", r_var), ("forgery", r_forge)]:
    print(f"{name:18s} final={r['final_score']:.3f} verdict={r['verdict']:8s} reasons={r['reasons']}")

assert r_same["final_score"] > r_forge["final_score"] + 0.15, "same should score clearly above forgery"
assert r_var["final_score"] > r_forge["final_score"], "variation should score above forgery"

# ---- underline ----------------------------------------------------------------------
ul = draw_signature(0, underline=True)
r_ul_same = score(ul, draw_signature(0, underline=True, jitter=0.6, noise=8))
r_ul_missing = score(ul, genuine)
print("underline same   :", r_ul_same["feature_scores"]["underline"], r_ul_same["details"]["underlines"])
print("underline missing:", r_ul_missing["feature_scores"]["underline"], r_ul_missing["details"]["underlines"],
      [x for x in r_ul_missing["reasons"] if "nderline" in x])
assert r_ul_same["feature_scores"]["underline"] > 0.8
assert r_ul_same["details"]["underlines"] == [1, 1]
assert r_ul_missing["feature_scores"]["underline"] == 0.0
assert r_ul_missing["details"]["underlines"][1] == 0, "body-only signature must not report an underline"

# ---- Qwen word / letter blending ----------------------------------------------------
from signatures.compare import add_text_score, is_legible
assert is_legible("Ashok Kumar") and not is_legible("ILLEGIBLE") and not is_legible("")
base = score(genuine, same_noisy)
same_txt = add_text_score(dict(base), "Ashok Kumar", "Ashok Kumar")
diff_txt = add_text_score(dict(base), "Ashok Kumar", "Rahul Singh")
none_txt = add_text_score(dict(base), "ILLEGIBLE", "Ashok Kumar")
print("text same:", same_txt["text_score"], same_txt["final_score"],
      "| text different:", diff_txt["text_score"], diff_txt["final_score"],
      "| illegible:", none_txt["text_score"], none_txt["final_score"])
assert same_txt["final_score"] > diff_txt["final_score"]
assert none_txt["text_score"] is None and none_txt["final_score"] == base["final_score"]
print(same_txt["text_comparison"]["letter_changes"], diff_txt["text_comparison"]["letter_changes"][:2])
print("SMOKE TEST PASSED")
