"""
PINN Solver for 2D Acoustic Wave Equation
==========================================
Uses DeepXDE to solve: d2u/dt2 = c^2 * (d2u/dx2 + d2u/dz2)

Standard fully-connected network with tanh activation.
Trained by minimising PDE residual + boundary + initial conditions.

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import deepxde as dde
import numpy as np
import torch
import time
import os

# Force MPS (Apple Silicon GPU) if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")


def pinn_solve(
    f0=10.0,
    c=2000.0,
    domain_size=1280.0,    # metres (128 grid points * 10m spacing)
    t_max=0.3,             # seconds
    n_domain=10000,        # collocation points inside domain
    n_boundary=2000,       # boundary condition points
    n_initial=2000,        # initial condition points
    n_test=5000,           # test points
    layers=[3, 64, 64, 64, 64, 1],  # network architecture
    epochs=10000,          # training iterations
    lr=1e-3,               # learning rate
    verbose=True
):
    """
    Train a PINN to solve the acoustic wave equation.
    
    Input: (x, z, t) coordinates
    Output: u(x, z, t) — displacement field
    """

    if verbose:
        print(f"\n{'='*50}")
        print(f"PINN SOLVER — f0={f0}Hz")
        print(f"{'='*50}")

    # ─────────────────────────────────────
    # 1. DEFINE GEOMETRY AND TIME DOMAIN
    # ─────────────────────────────────────
    # Spatial domain: [0, domain_size] x [0, domain_size]
    # Time domain: [0, t_max]
    # Combined: 3D box in (x, z, t)

    geom = dde.geometry.Rectangle(
        xmin=[0, 0],
        xmax=[domain_size, domain_size]
    )
    timedomain = dde.geometry.TimeDomain(0, t_max)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # ─────────────────────────────────────
    # 2. DEFINE PDE RESIDUAL
    # ─────────────────────────────────────
    def wave_equation(x, u):
        """
        PDE residual: d2u/dt2 - c^2 * (d2u/dx2 + d2u/dz2) = 0
        x: input tensor [x, z, t]
        u: network output
        """
        du_xx = dde.grad.hessian(u, x, i=0, j=0)  # d2u/dx2
        du_zz = dde.grad.hessian(u, x, i=1, j=1)  # d2u/dz2
        du_tt = dde.grad.hessian(u, x, i=2, j=2)  # d2u/dt2

        return du_tt - c**2 * (du_xx + du_zz)

    # ─────────────────────────────────────
    # 3. INITIAL CONDITIONS
    # ─────────────────────────────────────
    # u(x,z,0) = Gaussian pulse at centre
    # du/dt(x,z,0) = 0

    src_x = domain_size / 2
    src_z = domain_size / 4

    def ic_u(x):
        """Initial displacement — Gaussian pulse at source."""
        r2 = (x[:, 0:1] - src_x)**2 + (x[:, 1:2] - src_z)**2
        sigma = domain_size / 20.0
        return np.exp(-r2 / (2 * sigma**2))

    def ic_ut(x):
        """Initial velocity — zero."""
        return np.zeros((len(x), 1))

    ic_displacement = dde.icbc.IC(
        geomtime, ic_u, lambda x, on: on, component=0
    )

    # ─────────────────────────────────────
    # 4. BOUNDARY CONDITIONS
    # ─────────────────────────────────────
    # Zero displacement at all boundaries (Dirichlet)

    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda x, on: on
    )

    # ─────────────────────────────────────
    # 5. BUILD AND TRAIN
    # ─────────────────────────────────────
    data = dde.data.TimePDE(
        geomtime,
        wave_equation,
        [ic_displacement, bc],
        num_domain=n_domain,
        num_boundary=n_boundary,
        num_initial=n_initial,
        num_test=n_test
    )

    # Network architecture
    net = dde.nn.FNN(
        layers,
        "tanh",
        "Glorot normal"
    )

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=lr,
        loss_weights=[1, 100, 10]  # PDE, IC, BC weights
    )

    # ─────────────────────────────────────
    # 6. TRAINING
    # ─────────────────────────────────────
    t_start = time.perf_counter()

    if verbose:
        print(f"Training for {epochs} epochs...")

    losshistory, train_state = model.train(
        iterations=epochs,
        display_every=epochs // 10
    )

    runtime = time.perf_counter() - t_start

    # ─────────────────────────────────────
    # 7. EVALUATE
    # ─────────────────────────────────────
    # Sample the solution on a grid at t = t_max/2
    nx_eval = 64
    nz_eval = 64
    x_eval = np.linspace(0, domain_size, nx_eval)
    z_eval = np.linspace(0, domain_size, nz_eval)
    X, Z = np.meshgrid(x_eval, z_eval)
    t_eval = np.full(X.shape, t_max / 2)

    X_flat = np.column_stack([
        X.flatten(),
        Z.flatten(),
        t_eval.flatten()
    ])

    u_pred = model.predict(X_flat)
    u_grid = u_pred.reshape(nz_eval, nx_eval)
    max_amp = np.abs(u_grid).max()

    final_loss = losshistory.loss_train[-1]
    if hasattr(final_loss, '__iter__'):
        final_loss = sum(final_loss)

    if verbose:
        print(f"\nPINN completed in {runtime:.1f}s")
        print(f"f0={f0}Hz | Max amplitude: {max_amp:.4e}")
        print(f"Final loss: {float(sum(losshistory.loss_train[-1])):.4e}")

    return {
        'runtime': runtime,
        'max_amp': max_amp,
        'u_grid': u_grid,
        'f0': f0,
        'losshistory': losshistory,
        'model': model
    }


if __name__ == '__main__':
    # Test at 10Hz first — the middle frequency
    # This tells us baseline PINN performance
    result = pinn_solve(
        f0=10.0,
        epochs=5000,    # reduced for quick test
        verbose=True
    )

    print(f"\n{'='*50}")
    print(f"RESULT SUMMARY")
    print(f"{'='*50}")
    print(f"Frequency: 10Hz")
    print(f"PINN runtime: {result['runtime']:.1f}s")
    print(f"FD runtime (same problem): ~0.333s")
    print(f"Slowdown factor: {result['runtime']/0.333:.0f}x")
    print(f"Max amplitude: {result['max_amp']:.4e}")
