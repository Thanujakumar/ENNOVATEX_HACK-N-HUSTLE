**Estimation of UAV parameters (range, velocity, and direction of arrival) using monostatic sensing in an ISAC scenario, based on the 3GPP TR 38.901 Rel-19 (Section 7.9) channel model.**
 **Approach**
-3GPP TR 38.901 (Rel-19) Section 7.9 is used for channel modeling and UAV target Radar Cross Section (RCS) characteristics,       ensuring realism and compliance.
-Monostatic sensing geometry: transmitter and receiver co-located, sharing the same antenna array.
-Signal processing handles classical estimation (range, Doppler, angle).
-AI/ML modules refine these estimates (denoising, sub-bin regression, clutter rejection).
-This hybrid design gives us both interpretability and robustness under noisy/multipath-rich conditions.
**What makes it unique?**
-First-principles 38.901-compliant simulator that generates training data aligned with 3GPP standards.
-Hybrid AI architecture (not black-box only) → balances explainability + performance.
-End-to-end pipeline from waveform → channel → I/Q → RDA map → estimates with uncertainty metrics.
**Solution Architecture**
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
**Technical Stack**
=>We leverage open-source projects for simulation, signal processing, and AI:

1.Channel Modeling
-sionna: 3GPP 38.901 simulation library (baseline)
-Custom extensions for Rel-19 ISAC Sec. 7.9 (monostatic, UAV RCS tables)
2.Signal Processing
-NumPy / SciPy : FFTs, filtering, MUSIC/ESPRIT
-Matplotlib :visualization of range–Doppler–AoA maps
3.Machine Learning
-PyTorch: CNN/Transformer architectures for regression/classification
-scikit-learn: preprocessing, metrics
4.Experiment Management
-Hydra: config management
**Technical Architecture**
1.Simulator
-Generates Tx waveforms & Rx I/Q samples under 38.901 Rel-19 ISAC conditions
-Includes multipath, Doppler, UAV RCS, clutter
2.Signal Processing Pipeline
-OFDM demodulation, channel estimation
-FFTs for range & Doppler estimation
-MUSIC/ESPRIT for AoA
-CFAR for detection
3.AI Modules
-CNN-based denoiser on RDA cubes
-Regression head for sub-bin refinement of range/velocity/angle
-Optional classifier for UAV detection vs clutter
4.Tracking & Fusion
-Extended Kalman Filter for trajectory estimation
-Multi-target handling with JPDA
**Implementation Details**
1.Waveform: NR-OFDM with configurable numerology (FR1/FR2)
2.Antenna: ULA/UPA arrays with calibration support
3.Outputs:
-UAV range (m)
-Radial velocity (m/s)
-Direction of arrival (°)
-Uncertainty (σ values per estimate)
4.Dataset: Generated via channel simulator, with labels from ground truth
**Salient Features**
1.Standards-Compliant: 3GPP TR 38.901 Rel-19 ISAC Sec. 7.9 support (monostatic UAV sensing)
2.Hybrid Design: Signal processing + AI/ML for robustness
3.Scalable: Works across FR1 and FR2 frequencies
4.Uncertainty Estimates: Outputs confidence intervals along with predictions
5.Extensible: Easy to add new channel models, waveforms, or ML modules


