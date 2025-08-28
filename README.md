#!/usr/bin/env python3
"""
generate_data.py
Synthetic IQ dataset generator (toy simulator).
Generates complex IQ frames for multiple "UAV" samples with simple parametric model:
  s(t) = A * exp( j*(2*pi*fc*t + 2*pi*fd*t + phase) ) * steering(aoa)
Outputs:
  - data/sample_iq.npy : shape (N, num_rx, num_t) complex64
  - data/labels.npy    : shape (N, 3) -> [range_m, vel_mps, aoa_deg]
"""

import argparse
import os
import numpy as np

def simulate_uav_echo(num_rx, num_t, fc=3e9, fs=1e6, rng_m=100.0, vel=10.0, aoa_deg=10.0, snr_db=20.0):
    """
    Very simplified narrowband model:
    - simulate a complex exponential with Doppler shift proportional to velocity
    - steering vector across num_rx (ULA with half-wavelength spacing)
    - add AWGN
    """
    c = 3e8
    lam = c / fc
    # Doppler frequency (Hz) for radial velocity vel (m/s)
    fd = 2 * vel / lam  # factor 2 because monostatic round-trip approx (toy)
    t = np.arange(num_t) / fs
    phase = np.random.uniform(0, 2*np.pi)
    # baseband signal (complex sinusoid)
    s = np.exp(1j*(2*np.pi*fd*t + phase)).astype(np.complex64)  # shape (num_t,)
    # steering vector (ULA)
    aoa_rad = np.deg2rad(aoa_deg)
    element_pos = np.arange(num_rx)
    steering = np.exp(-1j * 2*np.pi * element_pos * (np.sin(aoa_rad) * (lam/2) / lam))
    steering = steering.astype(np.complex64)  # shape (num_rx,)
    # tile to RX x time
    iq = (steering[:, None] * s[None, :])
    # scale by 1/range^2 (toy RCS) and amplitude
    amplitude = 1.0 / (rng_m**2 + 1e-6)
    iq *= amplitude
    # add noise for given SNR
    sig_pow = np.mean(np.abs(iq)**2)
    snr_lin = 10**(snr_db/10)
    noise_pow = sig_pow / snr_lin
    noise = (np.sqrt(noise_pow/2) * (np.random.randn(*iq.shape) + 1j*np.random.randn(*iq.shape))).astype(np.complex64)
    iq_noisy = iq + noise
    return iq_noisy

def generate_dataset(out_dir="data", num_samples=1000, num_rx=8, num_t=256, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rngs = np.random.uniform(20, 800, size=(num_samples,))   # meters
    vels = np.random.uniform(-20, 20, size=(num_samples,))  # m/s (radial)
    aoas = np.random.uniform(-60, 60, size=(num_samples,))  # degrees
    snrs = np.random.uniform(5, 25, size=(num_samples,))
    data = np.zeros((num_samples, num_rx, num_t), dtype=np.complex64)
    labels = np.zeros((num_samples, 3), dtype=np.float32)
    for i in range(num_samples):
        iq = simulate_uav_echo(num_rx=num_rx, num_t=num_t, rng_m=rngs[i], vel=vels[i], aoa_deg=aoas[i], snr_db=snrs[i])
        data[i] = iq
        labels[i] = np.array([rngs[i], vels[i], aoas[i]], dtype=np.float32)
        if (i+1) % 100 == 0:
            print(f"[generate] {i+1}/{num_samples}")
    np.save(os.path.join(out_dir, "sample_iq.npy"), data)
    np.save(os.path.join(out_dir, "labels.npy"), labels)
    print(f"[✓] Saved {num_samples} samples to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic UAV IQ dataset (toy)")
    parser.add_argument("--scenario", type=str, default="urban_macro", help="scenario name (informational)")
    parser.add_argument("--num_samples", type=int, default=1000, help="number of samples to generate")
    parser.add_argument("--out_dir", type=str, default="data", help="output directory")
    parser.add_argument("--num_rx", type=int, default=8)
    parser.add_argument("--num_t", type=int, default=256)
    args = parser.parse_args()
    generate_dataset(out_dir=args.out_dir, num_samples=args.num_samples, num_rx=args.num_rx, num_t=args.num_t)
