# Geo2O-HMR Implementation Notes

This directory starts from a one-stage HMR codebase and adds the training-time
components described in the submitted manuscript.

Implemented entry points:

- `datasets/dense_o2o.py`
  - 2D-safe Mosaic for projected or pseudo-supervised datasets.
  - Geometry-aware Copy-Blend for full-3D synthetic datasets.
  - Scale-map target rebuilding after Dense O2O target construction.
- `datasets/multiple_datasets.py`
  - Wraps the HMR datasets with the Geo2O densification module.
  - Exposes `set_epoch()` so the final no-augmentation stage can be enabled.
- `models/criterion.py`
  - Adds HMR-adapted matchability-aware loss (MAL) for the confidence head.
- `configs/run/train_all.yaml`
  - Enables Geo2O by default with `mosaic_prob=0.5`, `copy_blend_prob=0.5`,
    and `conf_loss_type='mal'`.
  - Caps densified targets with `geo2o_cfg.max_targets` (50 by default,
    matching the copied one-stage model `num_queries` setting).

The current Copy-Blend implementation is intentionally modular: it keeps the
camera/depth/3D-label update path in place and uses a simple soft rectangular
crop as the foreground mask. When an offline person library with segmentation
masks is available, replace the crop/mask section in
`DenseO2OAugmentor._make_copy_blend_candidate()` and `_paste_candidate()` while
keeping the target-label update code.

Training uses the same command style as the original one-stage runner:

```bash
accelerate launch main.py --mode train --cfg train_all
```

To run the copied one-stage baseline inside this scaffold, set:

```yaml
conf_loss_type: 'focal'
geo2o_cfg:
  enabled: False
```
