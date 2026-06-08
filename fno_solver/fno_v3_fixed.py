"""
FNO v3 — Fixed High-Frequency Training
=======================================
Fix: Use absolute MSE loss instead of relative L2.
Prevents degenerate zero-prediction at high frequencies
where wavefield amplitude is small.

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import torch
import numpy as np
import time
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fd_solver'))
from fd_solver import fd_solve, velocity_homogeneous
from neuralop.models import FNO as FNOModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_data(n_samples=300, nx=64, nz=64, f0=10.0, verbose=True):
    """Generate training data with per-frequency amplitude scaling."""
    if verbose:
        print(f"Generating {n_samples} samples at f0={f0}Hz...")

    inputs = []
    outputs = []
    amplitudes = []

    for i in range(n_samples):
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(nz//8, nz//3)

        result = fd_solve(
            nx=nx, nz=nz,
            dx=20.0, dz=20.0,
            dt=0.001, nt=300,
            f0=f0,
            c_model=velocity_homogeneous(nz, nx),
            src_x=src_x, src_z=src_z,
            nb=10, verbose=False
        )

        yy, xx = np.mgrid[0:nz, 0:nx]
        r2 = (xx - src_x)**2 + (yy - src_z)**2
        src_map = np.exp(-r2 / 10.0).astype(np.float32)

        u = result['u_final'].astype(np.float32)
        amp = np.abs(u).max()
        amplitudes.append(amp)

        inputs.append(src_map)
        outputs.append(u)

        if verbose and (i+1) % 50 == 0:
            print(f"  {i+1}/{n_samples}")

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    global_max = np.max(amplitudes)

    # KEY FIX: normalise by global max amplitude
    # This keeps relative scale between samples
    # but prevents collapse at low-amplitude frequencies
    if global_max > 0:
        outputs_norm = outputs / global_max
    else:
        outputs_norm = outputs

    if verbose:
        print(f"Amplitude range: {np.min(amplitudes):.4e} to {np.max(amplitudes):.4e}")
        print(f"Global max: {global_max:.4e}")
        print(f"Normalised range: [{outputs_norm.min():.4f}, {outputs_norm.max():.4f}]")

    return inputs, outputs_norm, global_max


def train_fno_v3(inputs, outputs, epochs=200, verbose=True):
    """Train FNO with absolute MSE loss."""

    n_train = int(0.8 * len(inputs))

    X_train = torch.FloatTensor(inputs[:n_train]).unsqueeze(1).to(device)
    Y_train = torch.FloatTensor(outputs[:n_train]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(inputs[n_train:]).unsqueeze(1).to(device)
    Y_val = torch.FloatTensor(outputs[n_train:]).unsqueeze(1).to(device)

    if verbose:
        print(f"\nTraining: {n_train} samples, {len(inputs)-n_train} val")

    model = FNOModel(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        n_layers=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # KEY FIX: Use absolute MSE loss instead of relative L2
    # Relative L2 = ||pred - true|| / ||true||
    # At high frequencies ||true|| is tiny -> loss -> 1.0 trivially
    # Absolute MSE doesn't have this problem
    criterion = torch.nn.MSELoss()

    best_val = float('inf')
    best_state = None
    t_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0
        n_batches = 0

        for i in range(0, n_train, 16):
            idx = perm[i:i+16]
            xb = X_train[idx]
            yb = Y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)

            # Absolute MSE loss
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            # Report both MSE and relative L2
            val_mse = criterion(val_pred, Y_val).item()
            val_l2 = (torch.norm(val_pred - Y_val) /
                     (torch.norm(Y_val) + 1e-8)).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"MSE: {train_loss/n_batches:.6f} | "
                  f"Val MSE: {val_mse:.6f} | "
                  f"Val L2: {val_l2:.4f}")

    training_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    if verbose:
        print(f"\nTraining done in {training_time:.1f}s")

    return model, training_time


def benchmark_fno_v3(f0=20.0):
    """Run fixed FNO benchmark at given frequency."""

    nx, nz = 64, 64

    print(f"\n{'='*55}")
    print(f"FNO v3 (FIXED) BENCHMARK — f0={f0}Hz")
    print(f"{'='*55}")

    inputs, outputs, global_max = generate_data(
        n_samples=300, nx=nx, nz=nz, f0=f0
    )

    model, train_time = train_fno_v3(
        inputs, outputs, epochs=200
    )

    # Test on 10 new source locations
    inference_times = []
    l2_errors = []

    print("\nTesting on 10 new source locations...")
    for _ in range(10):
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(nz//8, nz//3)

        yy, xx = np.mgrid[0:nz, 0:nx]
        r2 = (xx - src_x)**2 + (yy - src_z)**2
        src_map = np.exp(-r2 / 10.0).astype(np.float32)
        x_in = torch.FloatTensor(src_map).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(x_in)
        inf_time = time.perf_counter() - t0
        inference_times.append(inf_time)

        # Denormalise prediction
        fno_u = pred.squeeze().cpu().numpy() * global_max

        # FD ground truth
        fd_result = fd_solve(
            nx=nx, nz=nz, dx=20.0, dz=20.0,
            dt=0.001, nt=300, f0=f0,
            c_model=velocity_homogeneous(nz, nx),
            src_x=src_x, src_z=src_z,
            nb=10, verbose=False
        )
        fd_u = fd_result['u_final']

        l2 = np.linalg.norm(fno_u - fd_u) / (np.linalg.norm(fd_u) + 1e-10)
        l2_errors.append(l2)

    avg_inf = np.mean(inference_times) * 1000
    avg_l2 = np.mean(l2_errors)

    # FD timing
    fd_times = []
    for _ in range(5):
        r = fd_solve(nx=nx, nz=nz, dx=20.0, dz=20.0,
                    dt=0.001, nt=300, f0=f0,
                    c_model=velocity_homogeneous(nz, nx),
                    src_x=nx//2, src_z=nz//4,
                    nb=10, verbose=False)
        fd_times.append(r['runtime'] * 1000)
    avg_fd = np.mean(fd_times)

    print(f"\n{'='*55}")
    print(f"FIXED FNO RESULTS — f0={f0}Hz")
    print(f"{'='*55}")
    print(f"Training time:    {train_time:.1f}s")
    print(f"Inference time:   {avg_inf:.2f}ms")
    print(f"FD solve time:    {avg_fd:.2f}ms")
    print(f"L2 relative error: {avg_l2:.4f}")

    if avg_l2 < 0.1:
        print(f"STATUS: WORKING — L2 error {avg_l2:.4f} < 0.1 threshold")
    else:
        print(f"STATUS: STILL FAILING — L2 error {avg_l2:.4f}")

    print(f"\nv2 (broken) L2 at {f0}Hz: 1.0000")
    print(f"v3 (fixed)  L2 at {f0}Hz: {avg_l2:.4f}")
    print(f"Improvement: {1.0/max(avg_l2, 1e-6):.1f}x")

    return {
        'f0': f0,
        'train_time': train_time,
        'avg_inference_ms': avg_inf,
        'avg_fd_ms': avg_fd,
        'avg_l2': avg_l2
    }


if __name__ == '__main__':
    # Test the fix at 20Hz — the first frequency that failed
    result = benchmark_fno_v3(f0=20.0)
