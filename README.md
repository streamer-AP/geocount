# GeoCount

**Calibration-Free Multi-View Crowd Counting via Visual Geometry Foundation Model**

> Replacing manual camera calibration with [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) for multi-view crowd counting.

## Motivation

Traditional multi-view crowd counting (MVCC) relies on precise camera calibration (intrinsics + extrinsics) to project multi-view features onto a ground plane. This introduces two pain points:

1. **High calibration cost** — requires placing calibration boards or manually marking correspondences
2. **Poor deployment flexibility** — any camera movement requires re-calibration

GeoCount leverages VGGT to automatically estimate camera poses from images alone, achieving **zero-calibration-cost** multi-view crowd counting while retaining the geometric advantages of calibrated methods.

## Project Structure

```
geocount/
├── README.md
├── requirements.txt
├── .python-version
├── scripts/
│   ├── step1_setup_env.sh          # Environment setup
│   ├── step2_download_data.sh      # Dataset download guide
│   ├── step3_parse_gt_calib.py     # Parse GT calibration (XML → numpy)
│   ├── step4_run_vggt.py           # Run VGGT inference
│   ├── step4_mock_vggt.py          # Mock VGGT (no GPU needed, for testing)
│   ├── step5_evaluate.py           # Evaluate camera parameter accuracy
│   ├── step6_visualize.py          # Generate comparison plots
│   └── run_stage1.sh               # Run full Stage 1 pipeline
├── utils/
│   ├── coord_transform.py          # Coordinate transforms & Sim(3) alignment
│   └── metrics.py                  # Evaluation metrics
├── data/                           # Datasets (not tracked)
│   ├── Wildtrack/
│   └── MultiviewX/
└── results/                        # Output (not tracked)
    ├── gt_calibrations/
    ├── vggt_predictions/
    ├── evaluation/
    └── figures/
```

## Stage 1: VGGT Feasibility Validation

The current focus is validating whether VGGT can accurately estimate camera parameters in multi-view surveillance scenarios.

### Pipeline

```
Step 1  Setup environment (Python 3.11 + PyTorch + VGGT)
Step 2  Download datasets (Wildtrack / MultiviewX)
Step 3  Parse GT calibration parameters from XML
Step 4  Run VGGT inference on synchronized multi-view images
Step 5  Evaluate: Sim(3) alignment + error metrics
Step 6  Visualize: bird's-eye camera plots, error bars, intrinsic comparison
```

### Quick Start

**1. Setup**

```bash
# Clone
git clone https://github.com/streamer-AP/geocount.git
cd geocount

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Prepare data**

```bash
bash scripts/step2_download_data.sh
```

- **Wildtrack**: Download manually from [EPFL CVLab](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) → `data/Wildtrack/`
- **MultiviewX**: Auto-cloned via the script → `data/MultiviewX/`

**3. Run (with GPU)**

```bash
bash scripts/run_stage1.sh wildtrack 0
```

**4. Run (without GPU, mock mode)**

```bash
python scripts/step4_mock_vggt.py --dataset wildtrack
python scripts/step5_evaluate.py --dataset wildtrack
python scripts/step6_visualize.py --dataset wildtrack
```

### Evaluation Metrics

| Metric | Excellent | Usable | Needs Work |
|--------|-----------|--------|------------|
| Rotation error | < 2° | 2–5° | > 5° |
| Position error (aligned) | < 0.2m | 0.2–0.5m | > 0.5m |
| Reprojection error | < 10px | 10–30px | > 30px |
| Intrinsic error | < 5% | 5–15% | > 15% |

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥ 12GB VRAM (24GB recommended, e.g. RTX 4090 / A5000)
- **VGGT**: ~1.2B parameters, ~4.5GB download, requires Ampere+ for bfloat16
- **Mock mode**: No GPU needed — generates synthetic camera parameters for pipeline testing

## Roadmap

- [x] **Stage 1** — VGGT feasibility validation
- [ ] **Stage 2** — Baseline construction (MVDet/3DROM with GT vs. VGGT params)
- [ ] **Stage 3** — Core method design (robust fusion / VGGT multi-output reuse / weak supervision)
- [ ] **Stage 4** — Experiments & paper

## Datasets

| Dataset | Views | Persons | Notes |
|---------|-------|---------|-------|
| [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) | 7 | 20–40 | Primary benchmark |
| [MultiviewX](https://github.com/hou-yz/MultiviewX) | 6 | ~40 | Synthetic |

## References

1. Wang et al. "VGGT: Visual Geometry Grounded Transformer." CVPR 2025 (Best Paper).
2. Hou et al. "Multiview Detection with Feature Perspective Transformation." ECCV 2020.
3. Qiu et al. "3D Random Occlusion and Multi-layer Projection for Deep Multi-camera Pedestrian Localization." ECCV 2022.
4. Zhang et al. "Calibration-Free Multi-view Crowd Counting." ECCV 2022.
5. Zhang et al. "WSCF-MVCC: Weakly-supervised Calibration-free Multi-view Crowd Counting." PRCV 2025.
6. Jiang et al. "CountFormer: Multi-View Crowd Counting Transformer." ECCV 2024.

## License

MIT
