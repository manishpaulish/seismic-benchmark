"""
FNO Solver for 2D Acoustic Wave Equation
=========================================
Fourier Neural Operator learns the solution operator:
velocity model + source -> wavefield at time t

Unlike PINNs, FNO:
- Learns from data (FD solutions)
- Predicts in milliseconds after training
- Handles high frequencies naturally

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import torch
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fd_solver'))
from fd_solver import fd_solve, velocity_homogeneous

from neuralop.models import FNO as FNOModel

# Use MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────
# 1. GENERATE TRAINING DATA USING FD
# ─────────────────────────────────────
def generate_training_data(
    n_samples=200,
    nx=64, nz=64,
    dx=20.0, dz=20.0,
    dt=0.001,
    nt=200,
    f0=10.0,
    verbose=True
):
    """
    Generate training data for FNO.
    Each sample: random source location -> wavefield snapshot
    Uses FD solver as ground truth.
    """
    if verbose:
        print(f"Generating {n_samples} training samples at f0={f0}Hz...")

    inputs = []   # source location maps
    outputs = []  # wavefield snapshots

    for i in range(n_samples):
        # Random source location
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(nz//8, nz//4)

        c_model = velocity_homogeneous(nz, nx, c=2000.0)

        result = fd_solve(
            nx=nx, nz=nz,
            dx=dx, dz=dz,
            dt=dt, nt=nt,
            f0=f0,
            c_model=c_model,
            src_x=src_x,
            src_z=src_z,
            nb=10,
            verbose=False
        )

        # Input: source location as Gaussian on grid
        src_map = np.zeros((nz, nx))
        for zi in range(nz):
            for xi in range(nx):
                r2 = (xi - src_x)**2 + (zi - src_z)**2
                src_map[zi, xi] = np.exp(-r2 / 8.0)

        inputs.append(src_map)
        outputs.append(result['u_final'])

        if verbose and (i+1) % 20 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    if verbose:
        print(f"Data shape: inputs={inputs.shape}, outputs={outputs.shape}")

    return inputs, outputs


# ─────────────────────────────────────
# 2. TRAIN FNO
# ─────────────────────────────────────
def train_fno(
    inputs, outputs,
    n_modes=12,
    hidden_channels=32,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    verbose=True
):
    """Train FNO on generated data."""

    n_train = int(0.8 * len(inputs))
    X_train = torch.FloatTensor(inputs[:n_train]).unsqueeze(1).to(device)
    Y_train = torch.FloatTensor(outputs[:n_train]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(inputs[n_train:]).unsqueeze(1).to(device)
    Y_val = torch.FloatTensor(outputs[n_train:]).unsqueeze(1).to(device)

    if verbose:
        print(f"\nTraining FNO: {n_train} train, {len(inputs)-n_train} val")
        print(f"Architecture: modes={n_modes}, channels={hidden_channels}")

    # Build FNO model
    model = FNOModel(
        n_modes=(n_modes, n_modes),
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_layers=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.5
    )
    criterion = torch.nn.MSELoss()

    t_start = time.perf_counter()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Mini-batch training
        idx = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = idx[i:i+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = Y_train[batch_idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if verbose and (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss/n_train:.6f} | "
                  f"Val: {val_loss:.6f}")

    training_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    if verbose:
        print(f"\nFNO training completed in {training_time:.1f}s")
        print(f"Best val loss: {best_val_loss:.6f}")

    return model, training_time


# ─────────────────────────────────────
# 3. FNO INFERENCE (PREDICTION)
# ─────────────────────────────────────
def fno_predict(model, src_x, src_z, nx, nz):
    """
    Predict wavefield for a given source location.
    This is the key metric — inference time vs FD solve time.
    """
    src_map = np.zeros((nz, nx), dtype=np.float32)
    for zi in range(nz):
        for xi in range(nx):
            r2 = (xi - src_x)**2 + (zi - src_z)**2
            src_map[zi, xi] = np.exp(-r2 / 8.0)

    x_input = torch.FloatTensor(src_map).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    t_start = time.perf_counter()
    with torch.no_grad():
        pred = model(x_input)
    inference_time = time.perf_counter() - t_start

    return pred.squeeze().cpu().numpy(), inference_time


# ─────────────────────────────────────
# 4. FULL BENCHMARK
# ─────────────────────────────────────
def fno_benchmark(f0=10.0, n_samples=200, epochs=100, verbose=True):
    """
    Full FNO benchmark at a given frequency.
    Returns training time, inference time, and accuracy vs FD.
    """
    nx, nz = 64, 64
    dx, dz = 20.0, 20.0

    print(f"\n{'='*50}")
    print(f"FNO BENCHMARK — f0={f0}Hz")
    print(f"{'='*50}")

    # Generate training data
    t_data_start = time.perf_counter()
    inputs, outputs = generate_training_data(
        n_samples=n_samples,
        nx=nx, nz=nz,
        dx=dx, dz=dz,
        f0=f0,
        verbose=verbose
    )
    data_time = time.perf_counter() - t_data_start
    print(f"Data generation: {data_time:.1f}s")

    # Train FNO
    model, training_time = train_fno(
        inputs, outputs,
        epochs=epochs,
        verbose=verbose
    )

    # Test inference — predict for a new source location
    src_x_test = nx // 2
    src_z_test = nz // 4

    pred_wavefield, inference_time = fno_predict(
        model, src_x_test, src_z_test, nx, nz
    )

    # Compare against FD ground truth
    fd_result = fd_solve(
        nx=nx, nz=nz,
        dx=dx, dz=dz,
        dt=0.001, nt=200,
        f0=f0,
        src_x=src_x_test,
        src_z=src_z_test,
        nb=10,
        verbose=False
    )

    fd_wavefield = fd_result['u_final']
    fd_time = fd_result['runtime']

    # L2 relative error
    l2_error = np.linalg.norm(pred_wavefield - fd_wavefield) / \
               (np.linalg.norm(fd_wavefield) + 1e-10)

    print(f"\n{'='*50}")
    print(f"FNO RESULTS — f0={f0}Hz")
    print(f"{'='*50}")
    print(f"Training time:    {training_time:.1f}s (one-time cost)")
    print(f"Inference time:   {inference_time*1000:.2f}ms")
    print(f"FD solve time:    {fd_time*1000:.2f}ms")
    print(f"Inference speedup vs FD: {fd_time/inference_time:.1f}x")
    print(f"L2 relative error: {l2_error:.4f}")
    print(f"FD max amplitude:  {np.abs(fd_wavefield).max():.4e}")
    print(f"FNO max amplitude: {np.abs(pred_wavefield).max():.4e}")

    return {
        'f0': f0,
        'training_time': training_time,
        'inference_time': inference_time,
        'fd_time': fd_time,
        'l2_error': l2_error,
        'speedup': fd_time / inference_time,
        'pred_wavefield': pred_wavefield,
        'fd_wavefield': fd_wavefield
    }


if __name__ == '__main__':
    # Quick test at 10Hz
    result = fno_benchmark(
        f0=10.0,
        n_samples=100,   # small for quick test
        epochs=50,       # small for quick test
        verbose=True
    )

    print(f"\nKey comparison:")
    print(f"PINN at 10Hz: 604s, amplitude wrong by 55,000x")
    print(f"FNO at 10Hz:  {result['inference_time']*1000:.2f}ms inference, "
          f"L2 error={result['l2_error']:.4f}")
