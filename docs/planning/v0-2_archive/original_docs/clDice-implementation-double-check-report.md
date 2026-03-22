# clDice Implementation Double-Check Report

**Date**: 2026-03-21
**Context**: Verifying that our `skeletonize()` call matches the original clDice paper exactly
**Triggered by**: Rule #30 compliance check — algorithms must match their literature definition

---

## Question

Our code changed from `skeletonize_3d()` to `skeletonize()` after scikit-image removed
the old function. Is this a silent algorithm change, or is it the same underlying code?

## Answer: Identical Algorithm — No Change

`skimage.morphology.skeletonize(3d_volume)` produces **bit-identical** results to the
old `skimage.morphology.skeletonize_3d(3d_volume)`. Both use the Lee (1994) 3D medial
surface thinning algorithm internally.

---

## Evidence

### 1. Original clDice Paper

[Shit et al. (2021). "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." CVPR.](https://arxiv.org/abs/2003.07311)

The paper uses **two different skeletonization approaches**:

- **Hard skeletonization (for the METRIC)**: Standard morphological thinning from
  scikit-image. The README notes "hard skeleton implementation using the skeletonize
  function from scikit-image."

- **Soft skeletonization (for the LOSS)**: Differentiable approximation using iterative
  min-pooling/max-pooling as proxies for morphological erosion/dilation (Algorithm 1,
  Section 4.1). Pure PyTorch — no scikit-image.

### 2. Official clDice Implementation

[github.com/jocpae/clDice](https://github.com/jocpae/clDice) — `cldice_metric/cldice.py`:

```python
from skimage.morphology import skeletonize, skeletonize_3d

def clDice(v_p, v_l):
    if len(v_p.shape)==2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
```

- **2D**: `skeletonize()` — Zhang (1984) algorithm
- **3D**: `skeletonize_3d()` — Lee (1994) algorithm

### 3. scikit-image `skeletonize_3d` Timeline

| scikit-image Version | Status |
|---------------------|--------|
| < 0.23 | `skeletonize_3d` available as standalone function |
| **0.23.1** (2024-04-10) | `skeletonize_3d` **deprecated** via [PR #7094](https://github.com/scikit-image/scikit-image/pull/7094) |
| **0.25.0** (2024-12-13) | `skeletonize_3d` **removed** entirely ([PR #7572](https://github.com/scikit-image/scikit-image/pull/7572)) |
| **0.26.0** (current) | `skeletonize_3d` does not exist — `ImportError` on import |

### 4. Algorithm Identity Proof

The new `skeletonize()` auto-detects 3D input and calls `_skeletonize_lee()` — the
**exact same** Lee (1994) algorithm:

```python
# scikit-image 0.26.0 source (skeletonize):
elif image.ndim == 3 or (image.ndim == 2 and method == 'lee'):
    skeleton = _skeletonize_lee(image)
```

`_skeletonize_lee` references: *"T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building
skeleton models via 3-D medial surface/axis thinning algorithms. Computer Vision,
Graphics, and Image Processing, 56(6):462-478, 1994."*

PR #7094 confirmed: `skeletonize_3d` was literally just a wrapper around
`_skeletonize_lee` with unnecessary uint8 scaling.

### 5. Output Difference

The **only** difference:

| Property | `skeletonize_3d()` (old) | `skeletonize()` (new) |
|----------|------------------------|---------------------|
| dtype | `np.uint8` | `np.bool_` |
| Foreground value | `255` | `True` |
| Background value | `0` | `False` |
| Algorithm | Lee (1994) | Lee (1994) |
| Topology | Identical | Identical |

For clDice metric computation (`np.sum(v * s) / np.sum(s)`), this dtype difference
has **zero effect** — both are truthy/falsy identically, and both cast correctly
to `np.float32` via `.astype(np.float32)`.

---

## Our Code

```python
# BEFORE (broken on scikit-image >= 0.25):
from skimage.morphology import skeletonize_3d
pred_skel = skeletonize_3d(pred_bin.astype(bool)).astype(np.float32)
label_skel = skeletonize_3d(label_bin.astype(bool)).astype(np.float32)

# AFTER (correct, identical algorithm):
from skimage.morphology import skeletonize
pred_skel = skeletonize(pred_bin.astype(bool)).astype(np.float32)
label_skel = skeletonize(label_bin.astype(bool)).astype(np.float32)
```

## Verdict

**Our fix is correct.** Using `skeletonize()` instead of `skeletonize_3d()` is NOT
an algorithm change — it is a function rename by scikit-image. The underlying Lee (1994)
algorithm is identical. No need to pin scikit-image to an older version.

This is different from the SWA mislabeling (`.claude/metalearning/2026-03-21-fake-swa-checkpoint-averaging-mislabeled.md`)
where we changed the actual algorithm. Here, the algorithm is preserved — only the
function name changed due to upstream API evolution.

---

---

## Cross-Reference: All Known clDice Implementations

### 1. jocpae/clDice — OFFICIAL (Shit et al., CVPR 2021)

**273 stars, 31 forks, last push 2024-02-14**

**Metric code** (`cldice_metric/cldice.py`):
```python
from skimage.morphology import skeletonize, skeletonize_3d

def cl_score(v, s):
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    if len(v_p.shape)==2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)
```

- **2D**: `skeletonize()` (Zhang-Suen 1984)
- **3D**: `skeletonize_3d()` (Lee 1994)
- **Dispatches on `len(v_p.shape)`**
- **No version pins** — no requirements.txt
- **BROKEN on scikit-image >= 0.25** — `skeletonize_3d` removed

**Loss code** (`cldice_loss/pytorch/soft_skeleton.py`):
- Pure PyTorch soft skeleton via iterative min/max pooling — no scikit-image
- Handles 2D (4D tensor) and 3D (5D tensor) via `len(img.shape)` dispatch
- `soft_erode`: separate-axis max_pool passes; `soft_dilate`: full max_pool

### 2. dmitrysarov/clDice — Third-party (OpenCV-based, 2D only)

**33 stars, 5 forks, last push 2020-01-10 (UNMAINTAINED)**

```python
import cv2

def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while(not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel
```

- **OpenCV-based skeletonization** — NOT scikit-image
- **2D ONLY** — `cv2.erode`/`cv2.dilate` are 2D operations, no 3D path
- **Different algorithm** — morphological thinning with cross structuring element
- Adds smoothing to cl_score numerator AND denominator (official has no smoothing)
- References earlier NeurIPS MedNeurIPS 2019 version, not CVPR 2021

### 3. cpuimage/clDice — TensorFlow reimplementation (2D only)

**0 stars, 0 forks, last push 2020-04-23 (UNMAINTAINED)**

- **Pure TensorFlow** — `tf.nn.erosion2d`, `tf.nn.dilation2d`
- **No scikit-image** — soft skeleton only, no hard metric
- **2D ONLY** — no 3D path
- Has adaptive convergence variant (not in official)
- Uses `divide_no_nan` for safety (official would div-by-zero on empty skeleton)

### 4. MONAI — Production Framework

**File**: `monai/losses/cldice.py` (adapted from jocpae/clDice)

- **LOSS ONLY, NO METRIC** — `monai.metrics` has no clDice
- Pure PyTorch soft skeleton, identical algorithm to official
- Handles 2D + 3D via `len(img.shape)` dispatch
- Default `iter_=3`
- Background exclusion via `exclude_background` flag

### 5. TorchMetrics — No clDice

TorchMetrics provides `DiceScore` and `GeneralizedDiceScore` but has **no clDice**.

### Summary Table

| Repo | Skeletonize (Metric) | Skeletonize (Loss) | 2D | 3D | scikit-image? | Broken on >=0.25? |
|------|---------------------|-------------------|----|----|--------------|------------------|
| **jocpae/clDice** (official) | `skeletonize` + `skeletonize_3d` | Soft (max-pool) | Yes | Yes | Yes | **YES** |
| **dmitrysarov/clDice** | `cv2.erode`/`cv2.dilate` | Soft (max_pool2d) | Yes | **No** | No | N/A |
| **cpuimage/clDice** | None (soft only) | Soft (tf.nn.erosion2d) | Yes | **No** | No | N/A |
| **MONAI** | **No metric** | Soft (max-pool) | Yes | Yes | No | N/A |
| **TorchMetrics** | **No clDice** | **No clDice** | — | — | — | — |

### Key Observation

Only the **official repo** (jocpae/clDice) provides a hard-skeleton clDice METRIC using
scikit-image. All other implementations provide only the differentiable soft-skeleton
LOSS. MONAI explicitly adapted the official code but only ported the loss, not the metric.

This means for **metric evaluation** (which is what the Analysis Flow needs), there is
exactly one reference implementation: `jocpae/clDice/cldice_metric/cldice.py`, which
uses `skeletonize_3d` for 3D volumes.

Since `skeletonize_3d` was removed in scikit-image 0.25, the correct replacement is
`skeletonize()` which calls the identical Lee (1994) algorithm internally.

---

## The Canonical clDice Formula

Consistent across ALL implementations:

```
tprec = cl_score(V_pred, skel(V_gt))   = |V_pred ∩ skel(V_gt)| / |skel(V_gt)|
tsens = cl_score(V_gt,   skel(V_pred)) = |V_gt ∩ skel(V_pred)| / |skel(V_pred)|
clDice = 2 × tprec × tsens / (tprec + tsens)
```

Where:
- **tprec** (Topology Precision): fraction of GT's skeleton covered by the prediction
- **tsens** (Topology Sensitivity): fraction of prediction's skeleton covered by the GT
- **clDice**: harmonic mean of both (identical structure to F1-score)

---

## References

1. [Shit et al. (2021). "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation." CVPR.](https://arxiv.org/abs/2003.07311)
2. [Official clDice implementation — github.com/jocpae/clDice](https://github.com/jocpae/clDice)
3. [dmitrysarov/clDice — OpenCV reimplementation](https://github.com/dmitrysarov/clDice)
4. [cpuimage/clDice — TensorFlow reimplementation](https://github.com/cpuimage/clDice)
5. [MONAI SoftclDiceLoss](https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/cldice.py)
6. [Lee, Kashyap, Chu (1994). "Building Skeleton Models via 3-D Medial Surface/Axis Thinning Algorithms." CVGIP 56(6):462-478.](https://doi.org/10.1006/cgip.1994.1042)
7. [Zhang, Suen (1984). "A Fast Parallel Algorithm for Thinning Digital Patterns." CACM.](https://doi.org/10.1145/357994.358023)
8. [scikit-image PR #7094 — deprecate skeletonize_3d](https://github.com/scikit-image/scikit-image/pull/7094)
9. [scikit-image PR #7572 — remove skeletonize_3d](https://github.com/scikit-image/scikit-image/pull/7572)
10. [scikit-image 0.23 release notes](https://scikit-image.org/docs/0.25.x/release_notes/release_0.23.html)
11. [scikit-image 0.25 release notes](https://scikit-image.org/docs/0.25.x/release_notes/release_0.25.html)
