# Geo2O-HMR

Efficient multi-person 3D Human Mesh Recovery via geometry-consistent Dense
One-to-One supervision.

Geo2O-HMR is a training-time framework for one-stage multi-person HMR. It keeps
the inference path of the underlying query-based mesh recovery model unchanged
and improves supervision density during training with:

- 2D-safe Mosaic for projected or pseudo-supervised data.
- Geometry-aware Copy-Blend for full-3D synthetic data.
- HMR-adapted matchability-aware loss (MAL) for confidence calibration.
- Scale-map target rebuilding after target densification.

The current implementation is scaffolded from an upstream one-stage HMR codebase
and adapts the Dense O2O / MAL ideas from DEIM for HMR targets.

## Project Status

Implemented:

- Scale-adaptive-token one-stage HMR training, evaluation, and inference
  pipeline.
- `datasets/dense_o2o.py` with Mosaic, Copy-Blend, scale-map rebuild, and
  `max_targets` guarding.
- `geo2o_cfg` in `configs/run/train_all.yaml`.
- MAL confidence loss via `conf_loss_type: 'mal'`.

Still expected to be refined:

- Copy-Blend currently uses a soft rectangular foreground crop. Replace this
  with an offline person library and segmentation masks when available.
- New Geo2O-HMR checkpoints are not included in this scaffold.

## Installation

Tested upstream environment:

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.1

Create an environment:

```bash
conda create -n geo2o-hmr python=3.11 -y
conda activate geo2o-hmr
```

Install PyTorch and xFormers:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

If `chumpy` raises compatibility errors, see:

```text
docs/fix_chumpy.md
```

## Weights

Prepare SMPL assets under:

```text
weights/smpl_data/smpl
```

Expected files include:

```text
body_verts_smpl.npy
J_regressor_h36m_correct.npy
SMPL_FEMALE.pkl
SMPL_MALE.pkl
SMPL_NEUTRAL.pkl
smpl_mean_params.npz
```

For training from scratch or fine-tuning, place DINOv2 ViT-B/14 pretrained
weights here:

```text
weights/dinov2/dinov2_vitb14_pretrain.pth
```

For Geo2O-HMR checkpoints, use:

```text
weights/geo2o_hmr
```

and update the corresponding `pretrain_path` in the run config. To evaluate an
upstream baseline checkpoint instead, point `pretrain_path` to that checkpoint
explicitly.

## Data Preparation

Follow the dataset preparation notes in:

```text
docs/data_preparation.md
```

Then run:

```bash
python debug_data.py
```

Visualizations will be saved to:

```text
datasets_visualization
```

## Training

Configure Accelerate once:

```bash
accelerate config
```

Train Geo2O-HMR on all configured datasets:

```bash
accelerate launch main.py --mode train --cfg train_all
```

The main Geo2O switches live in `configs/run/train_all.yaml`:

```yaml
conf_loss_type: 'mal'
mal_gamma: 1.5

geo2o_cfg:
  enabled: True
  mosaic_prob: 0.5
  copy_blend_prob: 0.5
  final_no_aug_epochs: 5
  max_targets: 50
```

To run the copied one-stage baseline without Geo2O densification:

```yaml
conf_loss_type: 'focal'
geo2o_cfg:
  enabled: False
```

Training logs and checkpoints are written to:

```text
outputs/logs
outputs/ckpts
```

## Inference

Run inference on the demo folder:

```bash
python main.py --mode infer --cfg demo
```

Distributed inference:

```bash
accelerate launch main.py --mode infer --cfg demo
```

Results are written to the `output_dir` configured in the selected run file.

## Evaluation

Examples:

```bash
python main.py --mode eval --cfg eval_ab
python main.py --mode eval --cfg eval_3dpw
python main.py --mode eval --cfg test_agora
```

Distributed evaluation:

```bash
accelerate launch main.py --mode eval --cfg eval_ab
```

## Code Map

- `datasets/dense_o2o.py`: Geo2O target densification.
- `datasets/multiple_datasets.py`: dataset wrapper that applies Geo2O training
  augmentation.
- `models/criterion.py`: HMR losses and MAL confidence loss.
- `models/geo2o_model.py`: one-stage HMR model backbone/decoder path.
- `configs/run/train_all.yaml`: default Geo2O training recipe.

## Acknowledgement

This scaffold reuses components from an upstream one-stage HMR implementation,
DINOv2, DAB-DETR, DINO, Accelerate, and DEIM. Please cite the relevant upstream
projects when using this code.
