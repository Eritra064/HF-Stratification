# HF-Stratification: Wearable ECG and IMU Pipeline for Heart Failure Risk Stratification

A signal processing and machine learning pipeline for heart failure (HF) risk stratification from a wrist-worn device that records single-lead ECG and 3-axis accelerometer (ACC) data. The current stage is motion-artifact removal: supervised denoising of Lead I ECG using the ACC signals as a motion reference. Risk stratification from the cleaned signals is the next stage.

> Work in progress. Developed as part of research in Dr. Zhishan Guo's group at NC State University.

## Stage 1: Motion-artifact removal

Wrist ECG is easily corrupted by arm and body motion. Because a wristband can also carry an IMU, the motion that causes the artifact is partly observable. This project trains a 1D convolutional network to take a corrupted ECG window plus the three ACC axes and output the clean waveform.

There is no public dataset with wrist ECG, wrist IMU, and clean reference ECG recorded together, so training pairs are built synthetically: clean Lead I from a clinical dataset is corrupted with artifacts generated from that subject's real accelerometer recordings.

## Method

**Artifact model** (`src/ecg_denoise/artifacts.py`). Each 5 s window is corrupted with:

1. An ACC-correlated component: a random linear mix of the three ACC axes (weights hidden from the model), passed through a random short FIR kernel, a nonlinearity, and a slow amplitude envelope.
2. An ACC-independent component: baseline wander (0.05 to 0.5 Hz) and high-frequency noise (20 to 120 Hz).

The total artifact is scaled to a random SNR between 0 and 12 dB. Training uses an expansive quadratic nonlinearity, while validation and test use a saturating tanh, so the model is evaluated on corruption it did not see during training.

**Model** (`src/ecg_denoise/model.py`). A 1D U-Net with three strided-convolution downsampling stages, a dilated-convolution bottleneck (dilation 2, 4, 8), and skip connections. Input: 4 channels (noisy ECG + 3 ACC axes). Output: clean ECG. Loss: MSE + 0.5 x L1.

**Evaluation.** Patient-level splits, so no subject appears in more than one of train, validation, and test. Metrics per test patient: RMSE (mV), Pearson r, and R² for corrupted input vs. denoised output.

## Data

This repo uses the SCG-RHC dataset from PhysioNet. The data is **not** included. Download it from PhysioNet under its license terms, convert each record to a CSV containing `ECG_lead_I`, `patch_ACC_lat`, `patch_ACC_hf`, and `patch_ACC_dv` (500 Hz), and place the files in `data/csv/` named like `TRM107-RHC1.csv`.

## Results

Held-out test patients, ACC reference enabled:

| Patient | Input RMSE (mV) | Output RMSE (mV) | Input r | Output r | Output R² |
|---|---|---|---|---|---|
| TRM107 | | | | | |
| TRM127 | | | | | |
| TRM170 | | | | | |
| **Mean** | | | | | |

**ACC ablation.** Same model with the ACC channels zeroed (`--no-imu`):

| Setting | Mean output RMSE (mV) | Mean output r |
|---|---|---|
| ECG + ACC | | |
| ECG only | | |

![Example denoising](figures/example_denoising.png)

**Efficiency** (`scripts/benchmark.py`, one 5 s window):

| Parameters | Size (float32) | Conv MACs | CPU latency (1 thread) |
|---|---|---|---|
| | | | |

## Usage

```bash
git clone https://github.com/<your-username>/hf-stratification.git
cd hf-stratification
pip install -e .

python scripts/run.py                      # train + evaluate (ECG + ACC)
python scripts/run.py --no-imu             # ablation
python scripts/benchmark.py                # size, compute, latency
python scripts/plot_example.py --patient TRM107 --start-sec 60
```

Settings live in `configs/default.yaml`. Command-line flags (`--csv-dir`, `--out-root`, `--epochs`, `--num-workers`, `--imu/--no-imu`) override it. Results are written to `outputs/denoise_imu/` or `outputs/denoise_noimu/`. A Colab version is in `notebooks/colab_demo.ipynb`.

## Repository structure

```
configs/default.yaml        settings and patient splits
src/ecg_denoise/
    artifacts.py            synthetic artifact model
    data.py                 loading, normalization, windowed dataset
    model.py                1D U-Net
    train.py                training loop
    evaluate.py             per-patient metrics and predictions
scripts/
    run.py                  train + evaluate
    benchmark.py            model size and latency
    plot_example.py         example figure
notebooks/colab_demo.ipynb
```

## Limitations and next steps

- Artifacts are synthetic. Validation on real free-living wrist recordings with a simultaneous reference ECG is still needed.
- The current cohort is small (13 subjects across splits); scaling to the full cohort is in progress.
- Next: on-device feasibility (quantization, smaller variants) and downstream cardiac risk stratification.

## Author

Sadia Rahman, sadia1704064@gmail.com
