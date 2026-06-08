"""
Finite Difference Solver for 2D Acoustic Wave Equation
=======================================================
Solves: d2u/dt2 = c(x,z)^2 * (d2u/dx2 + d2u/dz2)

Second-order accurate in space and time.
Explicit time-stepping (leap-frog scheme).
Ricker wavelet source.
Absorbing boundary conditions (sponge layer).

This is the reference solver. Every PINN and FNO result
gets compared against this. It must be correct.

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def ricker_wavelet(t, f0):
    tau = t - 1.0 / f0
    return (1.0 - 2.0 * (np.pi * f0 * tau)**2) * np.exp(-(np.pi * f0 * tau)**2)


def build_sponge(nx, nz, nb, alpha_max=0.02):
    damp = np.zeros((nz, nx))
    for i in range(nb):
        val = alpha_max * ((nb - i) / nb) ** 2
        damp[i, :]       += val
        damp[nz-1-i, :]  += val
        damp[:, i]       += val
        damp[:, nx-1-i]  += val
    return damp


def velocity_homogeneous(nz, nx, c=2000.0):
    return np.full((nz, nx), c, dtype=np.float64)


def velocity_layered(nz, nx, layer_velocities=None):
    if layer_velocities is None:
        layer_velocities = [1800, 2500, 3200, 4000]
    c = np.zeros((nz, nx), dtype=np.float64)
    boundaries = np.linspace(0, nz, len(layer_velocities) + 1, dtype=int)
    for k, vel in enumerate(layer_velocities):
        c[boundaries[k]:boundaries[k+1], :] = vel
    return c


def check_stability(dt, dx, dz, c_max):
    cfl = dt * c_max * np.sqrt(1/dx**2 + 1/dz**2)
    if cfl > 0.5:
        raise ValueError(
            f"CFL condition violated: {cfl:.3f} > 0.5\n"
            f"Reduce dt. Suggested dt_max = "
            f"{0.5 / (c_max * np.sqrt(1/dx**2 + 1/dz**2)):.6f} s"
        )
    print(f"CFL number: {cfl:.4f} (stable)")
    return cfl


def fd_solve(
    nx=256, nz=256,
    dx=10.0, dz=10.0,
    dt=0.001,
    nt=1000,
    f0=10.0,
    c_model=None,
    src_x=None,
    src_z=None,
    nb=30,
    verbose=True
):
    if c_model is None:
        c_model = velocity_homogeneous(nz, nx)

    if src_x is None:
        src_x = nx // 2
    if src_z is None:
        src_z = nz // 4

    c_max = c_model.max()
    check_stability(dt, dx, dz, c_max)

    damp = build_sponge(nx, nz, nb)

    u_prev = np.zeros((nz, nx))
    u_curr = np.zeros((nz, nx))
    u_next = np.zeros((nz, nx))

    snap_interval = max(1, nt // 10)
    snapshots = []
    snap_times = []

    t_vec = np.arange(nt) * dt
    t_start = time.perf_counter()

    for it in range(nt):
        d2u_dx2 = (np.roll(u_curr, -1, axis=1)
                   - 2 * u_curr
                   + np.roll(u_curr,  1, axis=1)) / dx**2

        d2u_dz2 = (np.roll(u_curr, -1, axis=0)
                   - 2 * u_curr
                   + np.roll(u_curr,  1, axis=0)) / dz**2

        u_next = (2 * u_curr - u_prev
                  + dt**2 * c_model**2 * (d2u_dx2 + d2u_dz2))

        u_next *= (1 - damp)
        u_curr *= (1 - damp)

        u_next[src_z, src_x] += dt**2 * ricker_wavelet(t_vec[it], f0)

        u_prev = u_curr.copy()
        u_curr = u_next.copy()

        if it % snap_interval == 0:
            snapshots.append(u_curr.copy())
            snap_times.append(t_vec[it])
            if verbose:
                print(f"  Step {it:4d}/{nt} | t={t_vec[it]:.3f}s | "
                      f"max|u|={np.abs(u_curr).max():.4e}")

    runtime = time.perf_counter() - t_start

    if verbose:
        print(f"\nFD solver completed in {runtime:.3f}s")
        print(f"Grid: {nx}x{nz} | Steps: {nt} | f0: {f0}Hz")

    return {
        'snapshots': snapshots,
        'snap_times': snap_times,
        'u_final': u_curr,
        'runtime': runtime,
        'nx': nx, 'nz': nz,
        'dx': dx, 'dz': dz,
        'dt': dt, 'nt': nt,
        'f0': f0,
        'c_model': c_model
    }


def plot_snapshots(result, save_path=None):
    snaps = result['snapshots']
    times = result['snap_times']
    n = min(4, len(snaps))

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        idx = i * (len(snaps) // n)
        vmax = np.abs(snaps[idx]).max() * 0.5
        if vmax == 0:
            vmax = 1e-10
        im = ax.imshow(snaps[idx], cmap='seismic',
                       vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f't = {times[idx]:.3f}s')
        ax.set_xlabel('x (grid)')
        ax.set_ylabel('z (grid)')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f'FD Wavefield Snapshots — f0={result["f0"]}Hz', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("FD SOLVER — QUICK VERIFICATION TEST")
    print("=" * 50)

    result = fd_solve(
        nx=128, nz=128,
        dx=10.0, dz=10.0,
        dt=0.0015,
        nt=300,
        f0=10.0,
        verbose=True
    )

    print(f"\nRuntime: {result['runtime']:.3f}s")
    print(f"Final wavefield max amplitude: {np.abs(result['u_final']).max():.4e}")

    plot_snapshots(result, save_path='results/fd_test_snapshots.png')

    print("\nFD solver verification complete.")
    print("If you see a wave expanding outward — it is working.")
