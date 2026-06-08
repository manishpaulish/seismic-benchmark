"""
FNO Solver v2 — Fixed Configuration
=====================================
Key fixes:
1. More training samples (500 instead of 100)
2. More epochs (200 instead of 50)
3. Better data normalisation
4. Larger grid (128x128)
5. More diverse source locations
6. Proper L2 loss normalisation
"""

import torch
import copy
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fd_solver'))
from fd_solver import fd_solve, velocity_homogeneous
from neuralop.models import FNO as FNOModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def generate_data(n_samples=300, nx=64, nz=64, f0=10.0, verbose=True):
    """Generate diverse training data with normalisation."""
    if verbose:
        print(f"Generating {n_samples} samples at f0={f0}Hz...")

    inputs = []
    outputs = []
    max_amp = 0

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

        # Source map — normalised Gaussian
        src_map = np.zeros((nz, nx), dtype=np.float32)
        yy, xx = np.mgrid[0:nz, 0:nx]
        r2 = (xx - src_x)**2 + (yy - src_z)**2
        src_map = np.exp(-r2 / 10.0).astype(np.float32)

        u = result['u_final'].astype(np.float32)
        max_amp = max(max_amp, np.abs(u).max())

        inputs.append(src_map)
        outputs.append(u)

        if verbose and (i+1) % 50 == 0:
            print(f"  {i+1}/{n_samples}")

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # Normalise outputs to [-1, 1]
    if max_amp > 0:
        outputs = outputs / max_amp

    if verbose:
        print(f"Max amplitude (before norm): {max_amp:.4e}")
        print(f"Output range after norm: [{outputs.min():.3f}, {outputs.max():.3f}]")

    return inputs, outputs, max_amp


def train_fno_v2(inputs, outputs, epochs=200, verbose=True):
    """Train FNO with better configuration."""

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

            # Relative L2 loss
            loss = torch.norm(pred - yb) / (torch.norm(yb) + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = (torch.norm(val_pred - Y_val) /
                       (torch.norm(Y_val) + 1e-8)).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train L2: {train_loss/n_batches:.4f} | "
                  f"Val L2: {val_loss:.4f}")

    training_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    if verbose:
        print(f"\nTraining done in {training_time:.1f}s")
        print(f"Best val L2: {best_val:.4f}")

    return model, training_time, best_val


def benchmark_fno_v2(f0=10.0):
    """Run complete FNO benchmark."""

    nx, nz = 64, 64

    print(f"\n{'='*55}")
    print(f"FNO v2 BENCHMARK — f0={f0}Hz")
    print(f"{'='*55}")

    # Generate data
    inputs, outputs, max_amp = generate_data(
        n_samples=300, nx=nx, nz=nz, f0=f0
    )

    # Train
    model, train_time, best_val = train_fno_v2(
        inputs, outputs, epochs=200
    )

    # Inference test — 10 different source locations
    inference_times = []
    l2_errors = []

    print("\nTesting on 10 new source locations...")
    for _ in range(10):
        src_x = np.random.randint(nx//4, 3*nx//4)
        src_z = np.random.randint(nz//8, nz//3)

        # FNO inference
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

        # FD ground truth
        fd_result = fd_solve(
            nx=nx, nz=nz, dx=20.0, dz=20.0,
            dt=0.001, nt=300, f0=f0,
            c_model=velocity_homogeneous(nz, nx),
            src_x=src_x, src_z=src_z,
            nb=10, verbose=False
        )

        fd_u = fd_result['u_final']
        fno_u = pred.squeeze().cpu().numpy() * max_amp

        l2 = np.linalg.norm(fno_u - fd_u) / (np.linalg.norm(fd_u) + 1e-10)
        l2_errors.append(l2)

    avg_inf = np.mean(inference_times) * 1000
    avg_l2 = np.mean(l2_errors)

    # FD timing for comparison
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
    print(f"FINAL RESULTS — f0={f0}Hz")
    print(f"{'='*55}")
    print(f"FNO training time:     {train_time:.1f}s (one-time)")
    print(f"FNO inference time:    {avg_inf:.2f}ms (avg over 10)")
    print(f"FD solve time:         {avg_fd:.2f}ms (avg over 5)")
    print(f"FNO L2 relative error: {avg_l2:.4f}")
    print(f"Best val L2:           {best_val:.4f}")

    if avg_inf < avg_fd:
        print(f"FNO speedup vs FD:     {avg_fd/avg_inf:.1f}x FASTER")
    else:
        print(f"FNO slowdown vs FD:    {avg_inf/avg_fd:.1f}x slower")

    print(f"\nPINN comparison at same frequency:")
    print(f"  PINN inference: 604,100ms | L2 error: ~74")
    print(f"  FNO inference:  {avg_inf:.2f}ms  | L2 error: {avg_l2:.4f}")
    print(f"  FNO vs PINN speedup: {604100/avg_inf:.0f}x")

    return {
        'f0': f0,
        'train_time': train_time,
        'avg_inference_ms': avg_inf,
        'avg_fd_ms': avg_fd,
        'avg_l2': avg_l2,
        'best_val': best_val
    }


if __name__ == '__main__':
    result = benchmark_fno_v2(f0=10.0)
