File: src/scripts/generate_data.py#!/usr/bin/env python3
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

File: src/scripts/run_sp_baseline.py
#!/usr/bin/env python3
"""
run_sp_baseline.py
Simple signal-processing baseline demonstration:
- Load IQ data
- Compute per-sample:
    - Range proxy via frequency-of-arrival (coarse using FFT across time)
    - Doppler proxy via short-time FFT (coarse)
    - AoA via simple Bartlett beamformer (steering and spatial FFT)
Outputs a CSV with baseline estimates and prints summary RMSE (vs labels) if labels exist.
"""

import argparse
import numpy as np
import os
import csv

def estimate_baseline(iq_sample, fs=1e6, fc=3e9):
    # iq_sample: shape (num_rx, num_t)
    num_rx, num_t = iq_sample.shape
    # Range proxy: use energy-weighted delay-less metric (toy) -> use phase slope to approximate Doppler -> map to velocity
    # Coarse Doppler estimate: FFT across time on the first RX
    s = iq_sample[0]
    S = np.fft.fftshift(np.fft.fft(s))
    freqs = np.fft.fftshift(np.fft.fftfreq(num_t, d=1/fs))
    peak_idx = np.argmax(np.abs(S))
    fd_est = freqs[peak_idx]  # in Hz (toy)
    # map fd to velocity (approx): fd = 2*v/lam  => v = fd*lam/2
    c = 3e8
    lam = c / fc
    v_est = fd_est * lam / 2.0
    # AoA: simple beamforming: compute FFT across RX
    rx_cov = np.mean(iq_sample * np.conj(iq_sample), axis=1)  # shape (num_rx,)
    # steering-like energy per angle
    angles = np.linspace(-90, 90, 181)
    energies = []
    for a in angles:
        a_rad = np.deg2rad(a)
        steering = np.exp(-1j * np.arange(num_rx) * np.sin(a_rad) * 2*np.pi * (lam/2) / lam)
        steering = steering.astype(np.complex64)
        energies.append(np.abs(np.vdot(steering, np.mean(iq_sample, axis=1)))**2)
    aoa_est = angles[int(np.argmax(energies))]
    # Range proxy: inverse energy heuristic (toy)
    power = np.mean(np.abs(iq_sample)**2)
    rng_est = 1.0 / (np.sqrt(power) + 1e-12)  # not physical; just a proxy for baseline demo
    return rng_est, float(v_est), float(aoa_est)

def run_baseline(input_file, out_csv="sp_baseline_results.csv"):
    data = np.load(input_file)  # shape (N, num_rx, num_t)
    N = data.shape[0]
    results = []
    for i in range(N):
        rng, vel, aoa = estimate_baseline(data[i])
        results.append((i, rng, vel, aoa))
        if (i+1) % 100 == 0:
            print(f"[baseline] {i+1}/{N}")
    # save CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "range_proxy", "vel_mps", "aoa_deg"])
        writer.writerows(results)
    print(f"[✓] Baseline results saved to {out_csv}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to data/sample_iq.npy")
    parser.add_argument("--out", type=str, default="sp_baseline_results.csv")
    args = parser.parse_args()
    run_baseline(args.input, args.out)
File: src/scripts/train_ai.py
#!/usr/bin/env python3
"""
train_ai.py
Simple PyTorch training script to regress [range, vel, aoa] from flattened IQ magnitude.
This is a toy model (ML baseline) demonstrating the pipeline.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

class IQDataset(Dataset):
    def __init__(self, data_path="data/sample_iq.npy", labels_path="data/labels.npy"):
        data = np.load(data_path)  # (N, num_rx, num_t)
        labels = np.load(labels_path)  # (N, 3)
        self.X = np.abs(data).astype(np.float32)  # magnitude features
        self.y = labels.astype(np.float32)
        self.N, self.RX, self.T = self.X.shape
        self.X = self.X.reshape(self.N, -1)  # flatten
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim, hidden=512, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train(cfg):
    dataset = IQDataset(data_path=cfg["data"]["data_path"], labels_path=cfg["data"]["labels_path"])
    train_loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRegressor(input_dim=dataset.X.shape[1], hidden=cfg["model"]["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.MSELoss()
    epochs = cfg["train"]["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(dataset)
        if (epoch+1) % max(1, epochs//5) == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pth")
    print("[✓] Model saved to checkpoints/model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
File: src/scripts/evaluate.py
#!/usr/bin/env python3
"""
evaluate.py
Loads the trained model and computes RMSE on the dataset.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from train_ai import IQDataset, SimpleRegressor

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets)**2, axis=0))

def evaluate(cfg, model_path):
    dataset = IQDataset(data_path=cfg["data"]["data_path"], labels_path=cfg["data"]["labels_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRegressor(input_dim=dataset.X.shape[1], hidden=cfg["model"]["hidden"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    rmse = compute_rmse(preds, trues)
    print(f"[✓] Evaluation RMSE (range_m, vel_mps, aoa_deg): {rmse}")
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.model)
File: src/configs/train.yaml
data:
  data_path: "data/sample_iq.npy"
  labels_path: "data/labels.npy"

train:
  lr: 0.001
  epochs: 30
  batch_size: 32

model:
  hidden: 512



