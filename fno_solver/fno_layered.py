"""
FNO Benchmark — Layered Velocity Model
=======================================
Tests whether FNO generalises from homogeneous
to layered media with reflections.

Critical experiment: if FNO fails here, it means
operator learning requires separate training per
velocity model — a major practical limitation.

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
from fd_solver import fd_solve
from layered_model import velocity_layered_realistic
from neuralop.models import FNO as FNOModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_layered_data(n_samples=300, nx=64, nz=64,
                          f0=10.0, verbose=True):
    """Generate training data using layered velocity model."""
    if verbose:
        print(f"Generating {n_samples} layered samples at f0={f0}Hz...")

    c_model_full = velocity_layered_realistic(nz, nx)
    inputs = []
    outputs = []
    amplitudes = []

    for i in range(n_samples):
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(2, nz//4)  # source in top layer

        result = fd_solve(
            nx=nx, nz=nz,
            dx=20.0, dz=20.0,
            dt=0.0008, nt=300,
            f0=f0,
            c_model=c_model_full,
            src_x=src_x, src_z=src_z,
            nb=10, verbose=False
        )

        yy, xx = np.mgrid[0:nz, 0:nx]
        r2 = (xx - src_x)**2 + (yy - src_z)**2
        src_map = np.exp(-r2 / 10.0).astype(np.float32)

        u = result['u_final'].astype(np.float32)
        amplitudes.append(np.abs(u).max())
        inputs.append(src_map)
        outputs.append(u)

        if verbose and (i+1) % 50 == 0:
            print(f"  {i+1}/{n_samples}")

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    global_max = np.max(amplitudes)

    if global_max > 0:
        outputs_norm = outputs / global_max
    else:
        outputs_norm = outputs

    if verbose:
        print(f"Amplitude range: {np.min(amplitudes):.4e}"
              f" to {np.max(amplitudes):.4e}")
        print(f"Global max: {global_max:.4e}")

    return inputs, outputs_norm, global_max


def train_fno_layered(inputs, outputs, epochs=200, verbose=True):
    """Train FNO on layered model data."""
    n_train = int(0.8 * len(inputs))

    X_train = torch.FloatTensor(inputs[:n_train]).unsqueeze(1).to(device)
    Y_train = torch.FloatTensor(outputs[:n_train]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(inputs[n_train:]).unsqueeze(1).to(device)
    Y_val = torch.FloatTensor(outputs[n_train:]).unsqueeze(1).to(device)

    if verbose:
        print(f"\nTraining FNO on layered data: {n_train} samples")

    model = FNOModel(
        n_modes=(16, 16),
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        n_layers=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
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
            xb, yb = X_train[idx], Y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
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
            val_mse = criterion(val_pred, Y_val).item()
            val_l2 = (torch.norm(val_pred - Y_val) /
                     (torch.norm(Y_val) + 1e-8)).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"MSE: {train_loss/n_batches:.6f} | "
                  f"Val L2: {val_l2:.4f}")

    training_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    if verbose:
        print(f"\nTraining done in {training_time:.1f}s")
    return model, training_time


def benchmark_fno_layered(f0=10.0):
    """Full FNO benchmark on layered model."""
    nx, nz = 64, 64
    c_model = velocity_layered_realistic(nz, nx)

    print(f"\n{'='*55}")
    print(f"FNO LAYERED MODEL BENCHMARK — f0={f0}Hz")
    print(f"{'='*55}")

    inputs, outputs, global_max = generate_layered_data(
        n_samples=300, nx=nx, nz=nz, f0=f0)

    model, train_time = train_fno_layered(inputs, outputs, epochs=200)

    # Test on 10 new source locations
    l2_errors = []
    inference_times = []

    print("\nTesting on 10 new source locations...")
    for _ in range(10):
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(2, nz//4)

        yy, xx = np.mgrid[0:nz, 0:nx]
        r2 = (xx - src_x)**2 + (yy - src_z)**2
        src_map = np.exp(-r2 / 10.0).astype(np.float32)
        x_in = torch.FloatTensor(src_map).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(x_in)
        inference_times.append(time.perf_counter() - t0)

        fno_u = pred.squeeze().cpu().numpy() * global_max
        fd_result = fd_solve(
            nx=nx, nz=nz, dx=20.0, dz=20.0,
            dt=0.0008, nt=300, f0=f0,
            c_model=c_model,
            src_x=src_x, src_z=src_z,
            nb=10, verbose=False)
        fd_u = fd_result['u_final']

        l2 = (np.linalg.norm(fno_u - fd_u) /
              (np.linalg.norm(fd_u) + 1e-10))
        l2_errors.append(l2)

    avg_inf = np.mean(inference_times) * 1000
    avg_l2 = np.mean(l2_errors)

    fd_times = []
    for _ in range(5):
        r = fd_solve(nx=nx, nz=nz, dx=20.0, dz=20.0,
                    dt=0.0008, nt=300, f0=f0,
                    c_model=c_model,
                    src_x=nx//2, src_z=nz//8,
                    nb=10, verbose=False)
        fd_times.append(r['runtime'] * 1000)
    avg_fd = np.mean(fd_times)

    print(f"\n{'='*55}")
    print(f"LAYERED FNO RESULTS — f0={f0}Hz")
    print(f"{'='*55}")
    print(f"Training time:     {train_time:.1f}s")
    print(f"Inference time:    {avg_inf:.2f}ms")
    print(f"FD solve time:     {avg_fd:.2f}ms")
    print(f"L2 relative error: {avg_l2:.4f}")
    print(f"\nHomogeneous FNO L2 at {f0}Hz for comparison:")
    homog = {5: 0.0091, 10: 0.0179}
    if f0 in homog:
        print(f"  Homogeneous: {homog[f0]:.4f}")
        print(f"  Layered:     {avg_l2:.4f}")
        if avg_l2 < homog[f0] * 3:
            print(f"  STATUS: FNO GENERALISES to layered media")
        else:
            print(f"  STATUS: FNO DEGRADES on layered media")

    return {
        'f0': f0,
        'train_time': train_time,
        'avg_inference_ms': avg_inf,
        'avg_fd_ms': avg_fd,
        'avg_l2': avg_l2
    }


if __name__ == '__main__':
    # Test at 5Hz and 10Hz — the two frequencies where FNO worked
    for f0 in [5, 10]:
        result = benchmark_fno_layered(f0=f0)
        print(f"\nf0={f0}Hz: L2={result['avg_l2']:.4f}, "
              f"inference={result['avg_inference_ms']:.2f}ms")
