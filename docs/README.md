# UAV Parameter Estimation with ISAC (Monostatic Sensing)

This repository contains our solution to **Problem Statement #7**:  
> *Estimation of UAV parameters (range, velocity, and direction of arrival) using monostatic sensing in an ISAC scenario, based on the 3GPP TR 38.901 Rel-19 (Section 7.9) channel model.*

---

## Approach

Our approach combines **standards-based channel modeling** with a **hybrid signal processing + AI pipeline**:

- **3GPP TR 38.901 (Rel-19) Section 7.9** is used for channel modeling and UAV target Radar Cross Section (RCS) characteristics, ensuring realism and compliance.  
- **Monostatic sensing geometry**: transmitter and receiver co-located, sharing the same antenna array.  
- **Signal processing** handles classical estimation (range, Doppler, angle).  
- **AI/ML** modules refine these estimates (denoising, sub-bin regression, clutter rejection).  
- This hybrid design gives us both **interpretability** and **robustness** under noisy/multipath-rich conditions.

**What makes it unique?**
- First-principles **38.901-compliant simulator** that generates training data aligned with 3GPP standards.  
- **Hybrid AI architecture** (not black-box only) → balances explainability + performance.  
- End-to-end pipeline from **waveform → channel → I/Q → RDA map → estimates** with uncertainty metrics.  

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

##  Technical Stack

We leverage **open-source projects** for simulation, signal processing, and AI:

- **Channel Modeling**  
  - [sionna](https://github.com/NVIDIA/sionna) – 3GPP 38.901 simulation library (baseline)  
  - Custom extensions for Rel-19 ISAC Sec. 7.9 (monostatic, UAV RCS tables)

- **Signal Processing**  
  - [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) – FFTs, filtering, MUSIC/ESPRIT  
  - [Matplotlib](https://matplotlib.org/) – visualization of range–Doppler–AoA maps

- **Machine Learning**  
  - [PyTorch](https://pytorch.org/) – CNN/Transformer architectures for regression/classification  
  - [scikit-learn](https://scikit-learn.org/) – preprocessing, metrics

- **Experiment Management**  
  - [Hydra](https://github.com/facebookresearch/hydra) – config management  
  - [Weights & Biases](https://wandb.ai/) (optional) – experiment tracking

---

## Technical Architecture

1. **Simulator**  
   - Generates Tx waveforms & Rx I/Q samples under 38.901 Rel-19 ISAC conditions  
   - Includes multipath, Doppler, UAV RCS, clutter  

2. **Signal Processing Pipeline**  
   - OFDM demodulation, channel estimation  
   - FFTs for range & Doppler estimation  
   - MUSIC/ESPRIT for AoA  
   - CFAR for detection  

3. **AI Modules**  
   - CNN-based denoiser on RDA cubes  
   - Regression head for sub-bin refinement of range/velocity/angle  
   - Optional classifier for UAV detection vs clutter  

4. **Tracking & Fusion**  
   - Extended Kalman Filter for trajectory estimation  
   - Multi-target handling with JPDA  

---

## Implementation Details

- **Waveform**: NR-OFDM with configurable numerology (FR1/FR2)  
- **Antenna**: ULA/UPA arrays with calibration support  
- **Outputs**:  
  - UAV range (m)  
  - Radial velocity (m/s)  
  - Direction of arrival (°)  
  - Uncertainty (σ values per estimate)  

- **Dataset**: Generated via channel simulator, with labels from ground truth  

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

---

## Technical Stack

We leverage **open-source projects** for simulation, signal processing, and AI:

- **Channel Modeling**  
  - [sionna](https://github.com/NVIDIA/sionna) – 3GPP 38.901 simulation library (baseline)  
  - Custom extensions for Rel-19 ISAC Sec. 7.9 (monostatic, UAV RCS tables)

- **Signal Processing**  
  - [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) – FFTs, filtering, MUSIC/ESPRIT  
  - [Matplotlib](https://matplotlib.org/) – visualization of range–Doppler–AoA maps

- **Machine Learning**  
  - [PyTorch](https://pytorch.org/) – CNN/Transformer architectures for regression/classification  
  - [scikit-learn](https://scikit-learn.org/) – preprocessing, metrics

- **Experiment Management**  
  - [Hydra](https://github.com/facebookresearch/hydra) – config management  
  - [Weights & Biases](https://wandb.ai/) (optional) – experiment tracking

---

##Technical Architecture

1. **Simulator**  
   - Generates Tx waveforms & Rx I/Q samples under 38.901 Rel-19 ISAC conditions  
   - Includes multipath, Doppler, UAV RCS, clutter  

2. **Signal Processing Pipeline**  
   - OFDM demodulation, channel estimation  
   - FFTs for range & Doppler estimation  
   - MUSIC/ESPRIT for AoA  
   - CFAR for detection  

3. **AI Modules**  
   - CNN-based denoiser on RDA cubes  
   - Regression head for sub-bin refinement of range/velocity/angle  
   - Optional classifier for UAV detection vs clutter  

4. **Tracking & Fusion**  
   - Extended Kalman Filter for trajectory estimation  
   - Multi-target handling with JPDA  

---

##Implementation Details

- **Waveform**: NR-OFDM with configurable numerology (FR1/FR2)  
- **Antenna**: ULA/UPA arrays with calibration support  
- **Outputs**:  
  - UAV range (m)  
  - Radial velocity (m/s)  
  - Direction of arrival (°)  
  - Uncertainty (σ values per estimate)  

- **Dataset**: Generated via channel simulator, with labels from ground truth  

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

