# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GeoCount is a calibration-free multi-view crowd counting system that uses VGGT (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) to automatically estimate camera parameters, eliminating manual calibration. The goal is to validate VGGT's viability as a drop-in replacement for manual calibration in multi-view crowd counting pipelines.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python version: 3.11 (see `.python-version`). VGGT requires ~4.5GB download from HuggingFace on first run. GPU with ≥12GB VRAM (CUDA) recommended; Apple Silicon (MPS) and CPU are also supported.

## Running the Pipeline

**Full pipeline** (requires GPU for real VGGT):
```bash
bash scripts/run_stage1.sh <dataset> <frame_id>
# Example:
bash scripts/run_stage1.sh multiviewx 0
```

**Mock mode** (CPU-only, no GPU needed, for testing steps 5–6):
```bash
python scripts/step4_mock_vggt.py --dataset multiviewx
python scripts/step5_evaluate.py --dataset multiviewx --frame_id 0
python scripts/step6_visualize.py --dataset multiviewx --frame_id 0
```

**Individual steps:**
```bash
python scripts/step3_parse_gt_calib.py --dataset multiviewx
python scripts/step4_run_vggt.py --dataset multiviewx --frame_id 0 --batch_size 4
python scripts/step5_evaluate.py --dataset multiviewx --frame_id 0
python scripts/step6_visualize.py --dataset multiviewx --frame_id 0
```

There are no automated tests. Validation is done by running the evaluation pipeline and checking `results/evaluation/` JSON outputs.

## Pipeline Architecture

The 6-step pipeline runs sequentially; each step produces files consumed by later steps:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 3 | `step3_parse_gt_calib.py` | Dataset XML calibration files | `results/gt_calibrations/{dataset}/gt_cameras.npz` |
| 4 | `step4_run_vggt.py` or `step4_mock_vggt.py` | Dataset images | `results/vggt_predictions/vggt_{dataset}_frame{id}.npz` |
| 5 | `step5_evaluate.py` | GT NPZ + VGGT NPZ | `results/evaluation/evaluation_{dataset}_frame{id}.json` |
| 6 | `step6_visualize.py` | GT NPZ + VGGT NPZ + evaluation JSON | `results/figures/{dataset}/*.png` |

## Key Architecture Decisions

**Coordinate System Alignment (Sim(3))**: VGGT outputs camera poses in an arbitrary coordinate frame. `step5_evaluate.py` applies Umeyama Sim(3) alignment (`utils/coord_transform.py:align_poses_sim3`) to register VGGT predictions to the GT coordinate system before computing metrics.

**Intrinsic Rescaling**: VGGT internally resizes images to 518×294 and outputs intrinsics for that resolution. `step5_evaluate.py` rescales focal lengths to the original image resolution (e.g., 1920×1080) via `rescale_intrinsics()` before comparing to GT. Without this, focal length error is ~71%; after rescaling it is ~7%.

**Device Selection** (`step4_run_vggt.py`): CUDA (with bfloat16 autocast) → MPS → CPU. Model weights stay in float32; inference uses autocast.

**Mock VGGT** (`step4_mock_vggt.py`): Generates GT cameras + Gaussian noise to simulate VGGT output, enabling full pipeline validation without a GPU. Hardcoded approximate camera positions for Wildtrack and MultiviewX.

## Datasets

- **MultiviewX**: Auto-downloadable synthetic dataset, 6 cameras, present in `data/MultiviewX/`
- **Wildtrack**: Real surveillance dataset (7 cameras, 1920×1080), must be downloaded manually; calibration XMLs in `calibrations/extrinsic/` and `calibrations/intrinsic_zero/`

## Evaluation Metrics & Thresholds

| Metric | Excellent | Usable | Needs Work |
|--------|-----------|--------|------------|
| Relative rotation error | < 2° | 2–5° | > 5° |
| Position error (Sim3-aligned) | < 0.2m | 0.2–0.5m | > 0.5m |
| Reprojection error | < 10px | 10–30px | > 30px |
| Focal length error | < 5% | 5–15% | > 15% |

## Project Stage

Currently in **Stage 1: VGGT Feasibility Validation**. The roadmap continues to Stage 2 (MVDet baseline with GT vs VGGT params), Stage 3 (core method design), and Stage 4 (paper submission targeting CVPR/ICCV/ECCV).
