#!/usr/bin/env python3
"""
Train and apply binary pose-classification models for Pilotless Poses.

Two subcommands:

  train   --pos DIR --neg DIR --out MODEL.joblib
          Reads every *_pose.json file in the positive and negative folders,
          turns each detected PERSON into one training sample (positive folder
          -> label 1, negative folder -> label 0), extracts generic pose
          features, trains a RandomForest, and saves it. A JSON metrics blob is
          printed to STDOUT; human-readable progress goes to STDERR.

  predict --model MODEL.joblib --threshold T  RESULT.json [RESULT.json ...]
          Scores each result file. An image's score is the MAX probability over
          its people (OR-combine): the image "passes" if ANY person scores at
          or above the threshold. Prints {resultPath: {prob, pass}} to STDOUT.

The pose JSON format is the one this app writes:
  { "image": "...", "people": [ { "keypoints": [ {name,x,y,confidence}, ... ] } ] }
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# COCO 17-keypoint order — the canonical, fixed feature ordering.
KEYPOINTS = [
    "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# A keypoint counts as "present" only above this confidence.
MIN_CONF = 0.1

# Feature-set schema version. Bump whenever the feature vector changes shape or
# meaning so models trained on an older set are rejected (with a clear "retrain"
# message) instead of crashing predict on a length mismatch.
#   v1 — 51 generic per-keypoint + 10 hand-elevation features (hands-up biased)
#   v2 — + 34 pose-agnostic features (joint angles, limb orientations, L/R
#        symmetry, leg/foot vertical relations, body bbox) so ANY pose is served
FEATURE_VERSION = 2


def log(msg):
    """Progress line on STDERR (streamed live to the app's Live Log)."""
    print(msg, file=sys.stderr, flush=True)


# ---- Feature engineering ----------------------------------------------------

def _build_feature_names():
    names = []
    # Per-keypoint: torso-normalized x, y, and confidence.
    for kp in KEYPOINTS:
        names.append(f"{kp}_nx")
        names.append(f"{kp}_ny")
        names.append(f"{kp}_conf")
    # A handful of explicit vertical relations (smaller y = higher up).
    # These 10 are arm/hand-elevation specific (the original "hands up" signal).
    names += [
        "left_wrist_minus_shoulder_ny",
        "right_wrist_minus_shoulder_ny",
        "left_elbow_minus_shoulder_ny",
        "right_elbow_minus_shoulder_ny",
        "left_wrist_minus_nose_ny",
        "right_wrist_minus_nose_ny",
        "left_hand_above_shoulder",
        "right_hand_above_shoulder",
        "both_hands_above_shoulder",
        "hands_above_nose",
    ]
    # ---- v2: pose-agnostic features (work for ANY pose) ---------------------
    # Joint flexion (cos of the interior angle): 1 = fully bent, -1 = straight.
    names += [
        "left_elbow_flex", "right_elbow_flex",
        "left_knee_flex", "right_knee_flex",
        "left_hip_flex", "right_hip_flex",
        "left_shoulder_flex", "right_shoulder_flex",
    ]
    # Limb-segment orientation: vertical component of the unit direction vector
    # (negative = the segment points upward). Generalizes "hand above shoulder".
    names += [
        "left_upper_arm_ny", "right_upper_arm_ny",
        "left_forearm_ny", "right_forearm_ny",
        "left_thigh_ny", "right_thigh_ny",
        "left_shin_ny", "right_shin_ny",
        "torso_ny", "torso_nx",
    ]
    # Left/right symmetry (|L - R| of the flexion features).
    names += [
        "elbow_flex_sym", "knee_flex_sym", "hip_flex_sym", "shoulder_flex_sym",
    ]
    # Generalized vertical relations for legs/feet (mirror the hand ones).
    names += [
        "left_ankle_minus_hip_ny", "right_ankle_minus_hip_ny",
        "left_knee_minus_hip_ny", "right_knee_minus_hip_ny",
        "left_wrist_minus_hip_ny", "right_wrist_minus_hip_ny",
        "left_foot_above_hip", "right_foot_above_hip", "any_foot_above_hip",
    ]
    # Whole-body shape.
    names += ["bbox_aspect", "vertical_spread", "horizontal_spread"]
    return names


FEATURE_NAMES = _build_feature_names()


def parse_person(person):
    """Return name -> (x, y, conf) for confident keypoints; low/absent -> None."""
    kps = {}
    for kp in (person or {}).get("keypoints", []):
        name = kp.get("name")
        if name is None:
            continue
        conf = float(kp.get("confidence", 0.0) or 0.0)
        if conf >= MIN_CONF:
            kps[name] = (float(kp.get("x", 0.0)), float(kp.get("y", 0.0)), conf)
        else:
            kps[name] = None
    return kps


def _center(kps):
    """Torso center: mid-shoulder if available, else mean of present points."""
    ls, rs = kps.get("left_shoulder"), kps.get("right_shoulder")
    if ls and rs:
        return (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    pts = [p for p in kps.values() if p]
    if not pts:
        return 0.0, 0.0
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _scale(kps, cx, cy):
    """Normalizing scale: shoulder-center to hip-center distance, with fallbacks."""
    ls, rs = kps.get("left_shoulder"), kps.get("right_shoulder")
    lh, rh = kps.get("left_hip"), kps.get("right_hip")
    if ls and rs and lh and rh:
        sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
        hx, hy = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0
        d = ((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5
        if d > 1.0:
            return d
    if ls and rs:
        d = ((ls[0] - rs[0]) ** 2 + (ls[1] - rs[1]) ** 2) ** 0.5
        if d > 1.0:
            return d
    # Fall back to the spread of present points.
    pts = [p for p in kps.values() if p]
    if len(pts) >= 2:
        spread = max(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts)
        if spread > 1.0:
            return spread
    return 1.0


def person_features(person):
    """Extract the fixed-length generic feature vector for one person.

    Returns None when the person has too few confident keypoints to be useful.
    """
    kps = parse_person(person)
    present = [name for name, p in kps.items() if p]
    if len(present) < 3:
        return None

    cx, cy = _center(kps)
    scale = max(_scale(kps, cx, cy), 1e-6)

    def ny(name):
        p = kps.get(name)
        return (p[1] - cy) / scale if p else None

    feats = []
    for kp in KEYPOINTS:
        p = kps.get(kp)
        if p:
            feats.append((p[0] - cx) / scale)
            feats.append((p[1] - cy) / scale)
            feats.append(p[2])
        else:
            feats.extend((0.0, 0.0, 0.0))

    # Vertical relations. Missing endpoints -> 0 (neutral).
    def rel(a, b):
        ya, yb = ny(a), ny(b)
        return (ya - yb) if (ya is not None and yb is not None) else 0.0

    lws = rel("left_wrist", "left_shoulder")
    rws = rel("right_wrist", "right_shoulder")
    les = rel("left_elbow", "left_shoulder")
    res = rel("right_elbow", "right_shoulder")
    lwn = rel("left_wrist", "nose")
    rwn = rel("right_wrist", "nose")
    left_up = 1.0 if lws < 0 else 0.0
    right_up = 1.0 if rws < 0 else 0.0
    feats += [
        lws, rws, les, res, lwn, rwn,
        left_up, right_up, left_up * right_up,
        1.0 if (lwn < 0 and rwn < 0) else 0.0,
    ]

    # ---- v2: pose-agnostic features -----------------------------------------
    # All of these are scale-invariant (cosines / unit-vector components /
    # normalized differences) and default to a neutral 0.0 when a needed
    # keypoint is absent, so the vector length stays fixed.
    def flex_cos(a, v, b):
        """Cos of the interior angle at vertex `v` (1 = bent, -1 = straight)."""
        pa, pv, pb = kps.get(a), kps.get(v), kps.get(b)
        if not (pa and pv and pb):
            return 0.0
        v1x, v1y = pa[0] - pv[0], pa[1] - pv[1]
        v2x, v2y = pb[0] - pv[0], pb[1] - pv[1]
        n1 = (v1x * v1x + v1y * v1y) ** 0.5
        n2 = (v2x * v2x + v2y * v2y) ** 0.5
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        c = (v1x * v2x + v1y * v2y) / (n1 * n2)
        return max(-1.0, min(1.0, c))

    def seg_dir(a, b):
        """Unit direction (dx/len, dy/len) from keypoint `a` to `b`."""
        pa, pb = kps.get(a), kps.get(b)
        if not (pa and pb):
            return 0.0, 0.0
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return 0.0, 0.0
        return dx / length, dy / length

    def midpoint(a, b):
        pa, pb = kps.get(a), kps.get(b)
        if pa and pb:
            return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
        if pa:
            return pa[0], pa[1]
        if pb:
            return pb[0], pb[1]
        return None

    # Joint flexion angles.
    le_flex = flex_cos("left_shoulder", "left_elbow", "left_wrist")
    re_flex = flex_cos("right_shoulder", "right_elbow", "right_wrist")
    lk_flex = flex_cos("left_hip", "left_knee", "left_ankle")
    rk_flex = flex_cos("right_hip", "right_knee", "right_ankle")
    lh_flex = flex_cos("left_shoulder", "left_hip", "left_knee")
    rh_flex = flex_cos("right_shoulder", "right_hip", "right_knee")
    ls_flex = flex_cos("left_hip", "left_shoulder", "left_elbow")
    rs_flex = flex_cos("right_hip", "right_shoulder", "right_elbow")

    # Limb-segment vertical orientation (only the y-component is salient here).
    _, lua_ny = seg_dir("left_shoulder", "left_elbow")
    _, rua_ny = seg_dir("right_shoulder", "right_elbow")
    _, lfa_ny = seg_dir("left_elbow", "left_wrist")
    _, rfa_ny = seg_dir("right_elbow", "right_wrist")
    _, lth_ny = seg_dir("left_hip", "left_knee")
    _, rth_ny = seg_dir("right_hip", "right_knee")
    _, lsh_ny = seg_dir("left_knee", "left_ankle")
    _, rsh_ny = seg_dir("right_knee", "right_ankle")

    # Torso axis (shoulder-center -> hip-center): both components (vertical = how
    # upright, horizontal = how much it leans).
    sc, hc = midpoint("left_shoulder", "right_shoulder"), midpoint("left_hip", "right_hip")
    if sc and hc:
        tdx, tdy = hc[0] - sc[0], hc[1] - sc[1]
        tlen = (tdx * tdx + tdy * tdy) ** 0.5
        torso_nx, torso_ny = (tdx / tlen, tdy / tlen) if tlen > 1e-6 else (0.0, 0.0)
    else:
        torso_nx = torso_ny = 0.0

    # Leg/foot vertical relations (mirror the hand ones; rel() handles missing).
    la_hip = rel("left_ankle", "left_hip")
    ra_hip = rel("right_ankle", "right_hip")
    lk_hip = rel("left_knee", "left_hip")
    rk_hip = rel("right_knee", "right_hip")
    lw_hip = rel("left_wrist", "left_hip")
    rw_hip = rel("right_wrist", "right_hip")
    left_foot_up = 1.0 if la_hip < 0 else 0.0
    right_foot_up = 1.0 if ra_hip < 0 else 0.0
    any_foot_up = 1.0 if (la_hip < 0 or ra_hip < 0) else 0.0

    # Whole-body bounding box over the present keypoints.
    pts = [p for p in kps.values() if p]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bw, bh = (max(xs) - min(xs)), (max(ys) - min(ys))
    bbox_aspect = (bw / bh) if bh > 1e-6 else 0.0

    feats += [
        le_flex, re_flex, lk_flex, rk_flex, lh_flex, rh_flex, ls_flex, rs_flex,
        lua_ny, rua_ny, lfa_ny, rfa_ny, lth_ny, rth_ny, lsh_ny, rsh_ny,
        torso_ny, torso_nx,
        abs(le_flex - re_flex), abs(lk_flex - rk_flex),
        abs(lh_flex - rh_flex), abs(ls_flex - rs_flex),
        la_hip, ra_hip, lk_hip, rk_hip, lw_hip, rw_hip,
        left_foot_up, right_foot_up, any_foot_up,
        bbox_aspect, bh / scale, bw / scale,
    ]
    return feats


# ---- Data loading -----------------------------------------------------------

def load_folder_samples(folder):
    """Yield one feature vector per person across all *_pose.json in `folder`."""
    files = sorted(glob.glob(os.path.join(folder, "*_pose.json")))
    if not files:
        # Be forgiving: also accept any *.json if no *_pose.json present.
        files = sorted(glob.glob(os.path.join(folder, "*.json")))
    samples = []
    n_images = 0
    n_skipped = 0
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            log(f"  ! could not read {os.path.basename(path)}: {e}")
            n_skipped += 1
            continue
        n_images += 1
        for person in data.get("people", []) or []:
            fv = person_features(person)
            if fv is not None:
                samples.append(fv)
    return samples, n_images, len(files), n_skipped


# ---- train ------------------------------------------------------------------

def cmd_train(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib

    log(f"[train] positive folder: {args.pos}")
    log(f"[train] negative folder: {args.neg}")

    pos, pos_imgs, pos_files, _ = load_folder_samples(args.pos)
    log(f"[train] positive: {len(pos)} people from {pos_imgs}/{pos_files} files")
    neg, neg_imgs, neg_files, _ = load_folder_samples(args.neg)
    log(f"[train] negative: {len(neg)} people from {neg_imgs}/{neg_files} files")

    if len(pos) == 0 or len(neg) == 0:
        raise SystemExit(
            "Need at least one usable person in BOTH folders "
            f"(positive={len(pos)}, negative={len(neg)}). "
            "Check the folders contain *_pose.json files with detected people."
        )

    X = np.array(pos + neg, dtype=np.float64)
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    log(f"[train] dataset: {X.shape[0]} samples x {X.shape[1]} features")

    # Stratified split when each class has enough samples; otherwise train on all.
    can_split = len(pos) >= 5 and len(neg) >= 5
    if can_split:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_tr, y_tr, X_te, y_te = X, y, X, y
        log("[train] few samples — evaluating on the training set")

    log("[train] fitting RandomForest…")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_tr, y_tr)

    train_acc = float(model.score(X_tr, y_tr))
    test_acc = float(model.score(X_te, y_te))
    log(f"[train] train accuracy: {train_acc:.1%}")
    log(f"[train] test accuracy:  {test_acc:.1%}")

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [
        {"name": FEATURE_NAMES[i], "importance": float(importances[i])}
        for i in top_idx
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "feature_version": FEATURE_VERSION,
            "min_conf": MIN_CONF,
        },
        args.out,
    )
    log(f"[train] saved model -> {args.out}")

    metrics = {
        "nPos": len(pos),
        "nNeg": len(neg),
        "nPosImages": pos_imgs,
        "nNegImages": neg_imgs,
        "trainAcc": train_acc,
        "testAcc": test_acc,
        "features": X.shape[1],
        "topFeatures": top_features,
    }
    # The final STDOUT line is the machine-readable result the server parses.
    print(json.dumps(metrics))


# ---- predict ----------------------------------------------------------------

def cmd_predict(args):
    import joblib

    bundle = joblib.load(args.model)
    model = bundle["model"]

    # Reject models trained on an older feature set rather than letting sklearn
    # raise a cryptic shape error: the feature vector this script now produces
    # would not match what the model expects.
    bundle_fv = bundle.get("feature_version", 1)
    if bundle_fv != FEATURE_VERSION:
        raise SystemExit(
            f"This model was trained with an older feature set (v{bundle_fv}); "
            f"the current feature set is v{FEATURE_VERSION}. Retrain the model."
        )

    out = {}
    for path in args.results:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        vectors = []
        for person in data.get("people", []) or []:
            fv = person_features(person)
            if fv is not None:
                vectors.append(fv)
        if not vectors:
            # No usable person -> cannot score; report prob 0, does not pass.
            out[path] = {"prob": 0.0, "pass": False, "scored": False}
            continue
        probs = model.predict_proba(np.array(vectors, dtype=np.float64))[:, 1]
        best = float(np.max(probs))
        out[path] = {
            "prob": best,
            "pass": bool(best >= args.threshold),
            "scored": True,
        }

    print(json.dumps(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="train a model from two folders")
    p_tr.add_argument("--pos", required=True, help="positive-examples folder")
    p_tr.add_argument("--neg", required=True, help="negative-examples folder")
    p_tr.add_argument("--out", required=True, help="output .joblib path")
    p_tr.set_defaults(func=cmd_train)

    p_pr = sub.add_parser("predict", help="score result json files")
    p_pr.add_argument("--model", required=True, help="trained .joblib path")
    p_pr.add_argument("--threshold", type=float, default=0.5)
    p_pr.add_argument("results", nargs="+", help="pose result json paths")
    p_pr.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
