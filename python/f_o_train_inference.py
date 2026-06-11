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
        {"model": model, "feature_names": FEATURE_NAMES, "min_conf": MIN_CONF},
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
