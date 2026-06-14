# Training an inference model — step-by-step

This is the full workflow for building a classifier (e.g. "skateboarding while
mid-air") from your scanned images. It keeps every step explicit so you stay in
control; nothing is hidden behind a single magic button.

The big idea: **processors and tags narrow a large image set down to clean
positive and negative folders, you Export those folders, then train a model on
them and verify it back in the preview.**

---

## 0. Scan images (once)

1. **Folder Scan** tab → pick a folder → **Start Scan**.
2. The **Live Log** overlay opens (toggle it any time from the header button) and
   shows VitPose running. Results land in `./pose_results` and are remembered, so
   re-scanning the same folder is a fast no-op (already-processed images are
   skipped).

---

## 1. Coarse filter with a processor

A *processor* is a tiny JavaScript predicate `o_img → keep (true) / drop (false)`
run over every image. Use it to cheaply narrow the set.

1. **Processors** tab → **New** → name it e.g. `hands_in_air` → write the
   predicate (use the built-in `hands_in_air` as a template) → **Save**.
2. **Image Preview** tab. Your processors appear in the **Pipeline** column.
   Toggle the ones you want **On**; drag to order them (active ones apply top to
   bottom — an image is kept only if it passes *all* active processors).
3. Click **Run Processors**. The kept images remain; the rest are filtered out.
   - **Invert** flips it: keep the images that do *not* match.
   - **Limit** caps the kept set to N images (handy for balanced training sets).
   - Results are **cached** (see "Caching" below) — re-running the same
     processor is instant.

---

## 2. Hand-pick the positives with a tag

The coarse filter is imperfect, so curate by hand.

1. In the sidebar, set **Active tag** to e.g. `skateboard_midair`.
2. Walk the kept images (arrow keys / click). For each good one, press **`m`**
   (or click the **★** on the row / the **Tag** toolbar button) to add that tag.
   Press again to remove it. An image can carry several tags
   (`o_img.a_s_tag` is the array).
3. The header badge shows **🏷 N** = how many images carry the active tag.

> Tip: to see *only* your tagged images, enable the built-in **`has_tag`**
> processor (set its **Tag** field in the Processor-controls panel to the same
> tag) and Run. Combined with `hands_in_air` it keeps "hands up **and** tagged".

---

## 3. Export the positive folder

1. In the **Export folder** box type e.g. `hands_in_the_air_skateboard`.
2. Click **Export N** — the kept images (after all filters/tags) and their pose
   JSON are copied server-side into `./hands_in_the_air_skateboard`. Idempotent;
   reports `Exported N` when done.

---

## 4. Build the negative folder

Same images, opposite selection — the "hands up but NOT skateboarding" set.

- Easiest: tag the negatives with a second tag (e.g. `skateboard_not`) as you go,
  then filter to that tag and Export to `hands_in_the_air_not_skateboard`.
- Or use **Invert** on a `has_tag:skateboard_midair` filter to get everything
  that passed the coarse filter but wasn't picked, then Export.

Aim for a few hundred of each, roughly balanced. Use **Limit** to cap.

---

## 5. Train the model

1. **Inference Models** tab → **New**.
2. Name it (e.g. `skateboard_midair`), set:
   - **Positive folder** → `hands_in_the_air_skateboard`
   - **Negative folder** → `hands_in_the_air_not_skateboard`
3. **Train**. Progress streams to the Live Log overlay. When done, the model
   shows **trained** with test accuracy + top features. Each successful train
   bumps the model's version.

> Works for **any** pose, not just arm/hand poses. Each detected person is turned
> into a pose-agnostic feature vector: scale-normalized keypoint positions, joint
> flexion angles (elbows, knees, hips, shoulders), limb-segment orientations,
> left/right symmetry, leg/foot vertical relations, and a body bounding box — so
> "leg raised", "sitting", "leaning", etc. train as well as "hands up".

> **Upgrading existing models:** the feature set is versioned (currently **v2**).
> A model trained on an older set can't be applied — **Apply** will report
> *"trained with an older feature set … Retrain the model."* Just hit **Train**
> again on the same folders to bring it up to date.

---

## 6. Verify in the preview

1. Back on **Image Preview**, the trained model appears in the **Inference
   Models** filter column.
2. Select it, set the **Threshold** slider, click **Apply**. Images scoring below
   the threshold are filtered out; the status shows `X passed, Y filtered`.
3. Tune the threshold and **Apply** again — this is **instant** (scores are
   cached; only retraining re-runs Python).
4. If it's good, Export the passing set, or feed it back as the next positive
   folder to refine.

---

## Caching (why re-applying is fast)

Running a processor or a model over every image is expensive, so results are
cached per **version**:

- **Processors** carry a `version` that bumps whenever you edit the **code**
  (toggling On/Off or renaming does *not* bump it). Running caches each
  processor's matched set; re-running the same processor is instant. Editing the
  code invalidates its cache.
- **Models** carry a `version` that bumps on every (re)train. Applying caches the
  per-image scores, so re-applying — even at a new threshold — skips Python.
  Retraining invalidates the cache.
- Caches are also keyed to the current image set, so scanning a different folder
  invalidates them.

You'll see `Done (cached)` / `(cached)` in the status when a cache hit is used.

---

## Quick reference

| Goal | Where |
|---|---|
| Scan a folder | Folder Scan → Start Scan |
| Write a filter | Processors → New |
| Enable/order filters | Preview → Pipeline column (On/Off, drag) |
| Run filters | Preview → Run Processors |
| Keep the non-matches | Preview → Invert |
| Cap the set | Preview → Limit |
| Hand-pick images | Preview → Active tag + `m` / ★ |
| Filter to a tag | enable `has_tag`, set its Tag field |
| Save a set to disk | Preview → Export folder + Export |
| Train | Inference Models → New → Train |
| Apply / verify | Preview → Inference Models column → Apply |
| Watch server output | Live Log button (header, any page) |
