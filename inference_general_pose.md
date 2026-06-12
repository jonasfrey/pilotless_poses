# Feature request: make AI inference equally good for ANY pose

## Goal
The inference training should classify **any** pose equally well — not just
"hands in the air". Today it works for any pose in principle (it's a data-driven
binary RandomForest over two folders), but the engineered features are biased
toward arm elevation, so non-arm poses (leg raised, sitting, leaning, crouching,
T-pose, twisting, etc.) only get the raw normalized keypoints and are
under-served.

## Current state (`python/f_o_train_inference.py`)
Per detected person the feature vector is **61 features**:
- **51 generic** — for each of the 17 COCO keypoints: torso-centered,
  shoulder→hip **scale-normalized** `x`, `y`, and `confidence`
  (`person_features`, ~L146-154). Translation/scale invariant; describes any pose.
- **10 hand-specific** engineered relations (`_build_feature_names`, ~L62-73):
  `*_wrist_minus_shoulder_ny`, `*_elbow_minus_shoulder_ny`, `*_wrist_minus_nose_ny`,
  `*_hand_above_shoulder`, `both_hands_above_shoulder`, `hands_above_nose`.

The 10 extra features dominate for hands-up but are neutral noise for other poses.
There are **no analogous engineered features for legs, torso, symmetry, or joint
angles**, which is why other poses aren't optimized.

## Proposed change
Add a **pose-agnostic engineered-feature block** so every pose gets strong,
directly-relevant signal. Keep the existing 10 hand features for backward-
compatible hands-up detection (retraining picks up the new ones automatically;
just bump model version / retrain existing models).

Candidate general features (all rotation/scale-normalized where sensible):
1. **Joint flexion angles** (most valuable): both elbows, both knees, both hips,
   both shoulders, plus neck/torso lean. Angle between the two adjacent limb
   segments at each joint. Pose-defining and viewpoint-robust.
2. **Left/right symmetry**: per-joint L vs R differences (angle + normalized
   position). Distinguishes symmetric vs one-sided poses (one leg up, one arm up).
3. **Limb-segment orientations**: angle of upper/lower arm, upper/lower leg, and
   torso axis relative to vertical. Generalizes the current "hand above shoulder"
   idea to every limb.
4. **Bounding-box aspect ratio** of present keypoints (wide vs tall pose) and
   overall vertical spread.
5. **Generalize the vertical-relation pattern** beyond hands: e.g.
   `ankle_minus_hip_ny`, `knee_minus_hip_ny`, `wrist_minus_hip_ny`, so "foot
   raised" gets the same treatment "hand raised" gets today.

## Implementation notes
- Contained change to `person_features()` and `_build_feature_names()` only; the
  fixed feature ordering + `FEATURE_NAMES` mechanism already carries through to
  metrics/top-features and the predict path unchanged.
- Missing keypoints must stay neutral (current convention: `0.0`), and angles
  need a defined value when an endpoint is absent (e.g. `0` + a present/absent
  flag) so the vector length stays fixed.
- Retraining required for existing models; bump model `version` so client score
  caches invalidate (server already keys cache on version).
- Consider documenting the new feature set in `docs/TRAINING.md`.

## Acceptance
- Train a non-arm pose (e.g. "leg raised", "sitting") from positive/negative
  folders and get test accuracy comparable to a hands-up model on similar data.
- Hands-up models still train and score as well as before (no regression).
