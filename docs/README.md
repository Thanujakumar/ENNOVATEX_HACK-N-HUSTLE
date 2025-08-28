# UAV Parameter Estimation with ISAC (Monostatic Sensing)

This repository contains our solution to **Problem Statement #7**:  
> *Estimation of UAV parameters (range, velocity, and direction of arrival) using monostatic sensing in an ISAC scenario, based on the 3GPP TR 38.901 Rel-19 (Section 7.9) channel model.*

---

## Approach

Our approach combines **standards-based channel modeling** with a **hybrid signal processing + AI pipeline**:

- **3GPP TR 38.901 (Rel-19) Section 7.9** for UAV sensing channel modeling.  
- **Monostatic sensing geometry** (Tx and Rx co-located).  
- **Signal processing** for range, Doppler, and AoA.  
- **AI/ML** modules for denoising, sub-bin regression, and clutter rejection.  
- **Hybrid design** = interpretable + robust.

**Unique points:**
- First-principles **38.901-compliant simulator**.  
- **Hybrid AI + SP design** (not just black-box ML).  
- End-to-end pipeline from waveform → estimates with **uncertainty metrics**.  

---

## Solution Architecture
 ┌─────────────────────┐
 │ 3GPP 38.901 Channel │
 │   (Rel-19, Sec 7.9) │
 └─────────┬───────────┘
           │
     Raw Rx I/Q
           │
┌──────────▼───────────┐
│ Signal Processing    │
│ - Range FFT          │
│ - Doppler FFT        │
│ - AoA Estimation     │
│ - CFAR Detection     │
└──────────┬───────────┘
           │ Features (RDA Cubes)
┌──────────▼───────────┐
│ AI / ML Modules      │
│ - CNN Denoiser       │
│ - Parameter Regressor│
│ - Clutter Classifier │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Tracking & Fusion    │
│ (Kalman/JPDA Filter) │
└──────────┬───────────┘
           │
 UAV Estimates: {Range, Velocity, AoA}

---

## Technical Stack

- **Channel Modeling**  
  - [sionna](https://github.com/NVIDIA/sionna) – 3GPP 38.901 simulation library  
  - Custom extensions for Rel-19 ISAC Sec. 7.9 (monostatic, UAV RCS tables)

- **Signal Processing**  
  - [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) – FFTs, filtering, MUSIC/ESPRIT  
  - [Matplotlib](https://matplotlib.org/) – visualization  

- **Machine Learning**  
  - [PyTorch](https://pytorch.org/) – CNN/Transformer architectures  
  - [scikit-learn](https://scikit-learn.org/) – preprocessing, metrics  

- **Experiment Management**  
  - [Hydra](https://github.com/facebookresearch/hydra) – config management  
  - [Weights & Biases](https://wandb.ai/) – optional tracking  

---

## Technical Architecture

1. **Simulator** – Generates Tx waveforms & Rx I/Q samples under 38.901 Rel-19 ISAC conditions.  
2. **Signal Processing Pipeline** – FFTs, MUSIC/ESPRIT, CFAR.  
3. **AI Modules** – CNN denoiser, regression head, classifier.  
4. **Tracking & Fusion** – EKF + JPDA for multi-target.  

---

## Implementation Details

- **Waveform**: NR-OFDM (FR1/FR2 supported)  
- **Antenna**: ULA/UPA arrays  
- **Outputs**: Range (m), Velocity (m/s), AoA (°), Uncertainty (σ)  
- **Dataset**: Simulator-generated, with ground-truth labels  

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/uav-isac.git
cd uav-isac

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
# Generate Dataset
python scripts/generate_data.py --scenario urban_macro --num_samples 10000

# Run Signal Processing Baseline
python scripts/run_sp_baseline.py --input data/sample_iq.npy

# Train AI Model
python scripts/train_ai.py --config configs/train.yaml

# Evaluate
python scripts/evaluate.py --model checkpoints/model.pth


