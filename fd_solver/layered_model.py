"""
Layered Velocity Model — Three-Way Benchmark
=============================================
Tests FD, PINN, FNO on a horizontally layered
velocity model with realistic crustal velocities.

This is the first step toward geological realism.
Introduces reflections and refractions not present
in the homogeneous medium.

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_solver import fd_solve

def velocity_layered_realistic(nz, nx):
    """
    4-layer model with realistic crustal velocities.
    
    Layer 1 (0-25%):   1800 m/s — sediment
    Layer 2 (25-50%):  2500 m/s — consolidated sediment  
    Layer 3 (50-75%):  3200 m/s — limestone
    Layer 4 (75-100%): 4000 m/s — basement rock
    
    These velocity contrasts produce strong reflections
    at each interface — the key difference from homogeneous.
    """
    c = np.zeros((nz, nx), dtype=np.float64)
    boundaries = [0, nz//4, nz//2, 3*nz//4, nz]
    velocities = [1800, 2500, 3200, 4000]
    
    for k, vel in enumerate(velocities):
        c[boundaries[k]:boundaries[k+1], :] = vel
    
    return c


def plot_velocity_model(c, save_path=None):
    """Visualise the layered velocity model."""
    plt.figure(figsize=(8, 6))
    im = plt.imshow(c/1000, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Velocity (km/s)')
    plt.title('Layered Velocity Model — 4 Crustal Layers')
    plt.xlabel('x (grid points)')
    plt.ylabel('z (grid points, depth)')
    plt.axhline(y=c.shape[0]//4, color='white', linestyle='--',
                alpha=0.7, label='Layer boundaries')
    plt.axhline(y=c.shape[0]//2, color='white', linestyle='--', alpha=0.7)
    plt.axhline(y=3*c.shape[0]//4, color='white', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def fd_layered_benchmark(frequencies=[5, 10, 20, 40, 80]):
    """Run FD benchmark on layered model."""
    
    print("=" * 60)
    print("FD BENCHMARK — LAYERED VELOCITY MODEL")
    print("4 layers: 1800/2500/3200/4000 m/s")
    print("=" * 60)
    
    nx, nz = 256, 256
    c_model = velocity_layered_realistic(nz, nx)
    
    print(f"\nVelocity model: {nz}x{nx} grid")
    print(f"Layer velocities: 1800/2500/3200/4000 m/s")
    print(f"Max velocity: {c_model.max():.0f} m/s\n")
    
    results = {}
    
    print(f"{'Freq (Hz)':<12} {'Runtime (s)':<15} {'Max Amp':<15}")
    print("-" * 45)
    
    for f0 in frequencies:
        result = fd_solve(
            nx=nx, nz=nz,
            dx=10.0, dz=10.0,
            dt=0.0008,  # smaller dt for higher velocity
            nt=500,
            f0=f0,
            c_model=c_model,
            src_x=nx//2,
            src_z=nz//8,  # shallow source in top layer
            nb=30,
            verbose=False
        )
        
        results[f0] = result
        print(f"{f0:<12} {result['runtime']:<15.3f} "
              f"{np.abs(result['u_final']).max():<15.4e}")
    
    print("-" * 45)
    print("\nCompare to homogeneous results:")
    print("Homogeneous: ~0.3s runtime, similar amplitude")
    print("Layered: reflections at layer boundaries expected")
    
    return results, c_model


if __name__ == '__main__':
    # Step 1: Visualise the velocity model
    nx, nz = 256, 256
    c_model = velocity_layered_realistic(nz, nx)
    plot_velocity_model(c_model, 
                       save_path='results/layered_velocity_model.png')
    
    # Step 2: Run FD benchmark
    results, c_model = fd_layered_benchmark()
    
    # Step 3: Plot wavefield snapshots at 10Hz
    # to show reflections from layer boundaries
    print("\nRunning detailed 10Hz simulation for wavefield visualisation...")
    result_10hz = fd_solve(
        nx=nx, nz=nz,
        dx=10.0, dz=10.0,
        dt=0.0008,
        nt=800,
        f0=10.0,
        c_model=c_model,
        src_x=nx//2,
        src_z=nz//8,
        nb=30,
        verbose=False
    )
    
    # Plot showing reflections
    snaps = result_10hz['snapshots']
    times = result_10hz['snap_times']
    n = min(4, len(snaps))
    
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for i, ax in enumerate(axes):
        idx = i * (len(snaps) // n)
        vmax = np.abs(snaps[idx]).max() * 0.3
        if vmax == 0:
            vmax = 1e-10
        im = ax.imshow(snaps[idx], cmap='seismic',
                      vmin=-vmax, vmax=vmax, aspect='auto')
        # Draw layer boundaries
        for boundary in [nz//4, nz//2, 3*nz//4]:
            ax.axhline(y=boundary, color='yellow',
                      linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f't = {times[idx]:.3f}s')
        ax.set_xlabel('x (grid)')
        ax.set_ylabel('z (grid, depth)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Wavefield in Layered Model — f0=10Hz\n'
                 '(yellow dashed = layer boundaries)', y=1.02)
    plt.tight_layout()
    plt.savefig('results/layered_wavefield_10hz.png',
               dpi=150, bbox_inches='tight')
    print("Saved layered wavefield plot")
    plt.show()
    
    print("\nLayered FD benchmark complete.")
    print("Key observation: reflections visible at layer boundaries")
    print("This is physically correct — harder for neural methods")
