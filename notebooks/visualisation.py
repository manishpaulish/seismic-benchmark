"""
Paper Figures — Seismic Benchmark
===================================
Figure 1: Wavefield comparison FD vs FNO at 5Hz
Figure 2: Training convergence PINN vs FNO
Figure 3: Summary bar chart — all methods all frequencies

Author: Manish Paul, IIT Kharagpur
Project: Seismic Benchmark Paper
Date: June 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fd_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fno_solver'))

from fd_solver import fd_solve, velocity_homogeneous
from fno_v2 import generate_data, train_fno_v2, device
import torch

# ── Style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'fd':   '#2196F3',   # blue
    'fno':  '#4CAF50',   # green
    'pinn': '#F44336',   # red
}

os.makedirs('results/figures', exist_ok=True)


# ══════════════════════════════════════════════════
# FIGURE 1: Wavefield comparison at 5Hz
# ══════════════════════════════════════════════════
def figure1_wavefield_comparison():
    print("Building Figure 1: Wavefield comparison...")

    nx, nz = 64, 64
    f0 = 5.0

    # FD ground truth
    fd_result = fd_solve(
        nx=nx, nz=nz, dx=20.0, dz=20.0,
        dt=0.001, nt=300, f0=f0,
        c_model=velocity_homogeneous(nz, nx),
        src_x=nx//2, src_z=nz//4,
        nb=10, verbose=False
    )
    fd_u = fd_result['u_final']

    # FNO prediction — train quickly
    print("  Training FNO for Figure 1...")
    inputs, outputs, global_max = generate_data(
        n_samples=200, nx=nx, nz=nz, f0=f0, verbose=False
    )
    model, _, _ = train_fno_v2(inputs, outputs, epochs=150, verbose=False)

    # FNO predict at same source location
    yy, xx = np.mgrid[0:nz, 0:nx]
    r2 = (xx - nx//2)**2 + (yy - nz//4)**2
    src_map = np.exp(-r2 / 10.0).astype(np.float32)
    x_in = torch.FloatTensor(src_map).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_in)
    fno_u = pred.squeeze().cpu().numpy() * global_max

    # Difference
    diff = fno_u - fd_u
    l2_error = np.linalg.norm(diff) / (np.linalg.norm(fd_u) + 1e-10)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    vmax = np.abs(fd_u).max() * 0.8

    im0 = axes[0].imshow(fd_u, cmap='seismic',
                          vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0].set_title('FD Solver (Reference)', color=COLORS['fd'],
                      fontweight='bold')
    axes[0].set_xlabel('x (grid)')
    axes[0].set_ylabel('z (grid, depth)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label='Displacement')

    im1 = axes[1].imshow(fno_u, cmap='seismic',
                          vmin=-vmax, vmax=vmax, aspect='auto')
    axes[1].set_title(f'FNO Prediction (L2={l2_error:.3f})',
                      color=COLORS['fno'], fontweight='bold')
    axes[1].set_xlabel('x (grid)')
    axes[1].set_ylabel('z (grid, depth)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8, label='Displacement')

    vmax_diff = np.abs(diff).max() * 0.8
    if vmax_diff == 0:
        vmax_diff = 1e-10
    im2 = axes[2].imshow(diff, cmap='RdBu_r',
                          vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
    axes[2].set_title(f'Difference (FNO - FD)', fontweight='bold')
    axes[2].set_xlabel('x (grid)')
    axes[2].set_ylabel('z (grid, depth)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8, label='Error')

    plt.suptitle(f'Wavefield Comparison — f0={f0}Hz, '
                 f'Homogeneous Medium\n'
                 f'FNO achieves {l2_error*100:.1f}% relative L2 error',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    path = 'results/figures/fig1_wavefield_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()
    return l2_error


# ══════════════════════════════════════════════════
# FIGURE 2: Training convergence comparison
# ══════════════════════════════════════════════════
def figure2_convergence():
    print("Building Figure 2: Training convergence...")

    # PINN loss history (from our experiments)
    # IC loss stuck at ~0.76 — the key finding
    pinn_steps = [0, 500, 1000, 1500, 2000, 2500,
                  3000, 3500, 4000, 4500, 5000]
    pinn_pde   = [1.49e8, 1.15e4, 3.56e3, 1.63e3, 9.07e2,
                  5.71e2, 3.90e2, 2.81e2, 2.96e3, 3.99e2, 1.49e2]
    pinn_ic    = [1.66e1, 7.60e-1, 7.52e-1, 7.58e-1, 7.63e-1,
                  7.64e-1, 7.64e-1, 7.64e-1, 7.62e-1, 7.65e-1, 7.64e-1]

    # FNO val L2 history (from our experiments)
    fno_epochs = [50, 100, 150, 200]
    fno_val_l2 = [0.0613, 0.0354, 0.0213, 0.0177]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: PINN loss components
    ax = axes[0]
    ax.semilogy(pinn_steps, pinn_pde,
                color=COLORS['pinn'], linewidth=2,
                label='PDE residual loss', linestyle='-')
    ax.semilogy(pinn_steps, pinn_ic,
                color='orange', linewidth=2,
                label='IC loss (stuck at ~0.76)', linestyle='--')
    ax.axhline(y=0.76, color='orange', linestyle=':',
               alpha=0.5, linewidth=1)
    ax.annotate('IC loss stagnates\n(spectral bias)',
                xy=(2500, 0.76), xytext=(1000, 0.3),
                fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange'))
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('PINN Training — f0=10Hz\n'
                 'IC loss stuck at 0.76 → spectral bias failure',
                 color=COLORS['pinn'])
    ax.legend(fontsize=9)
    ax.set_xlim(0, 5000)

    # Right: FNO convergence
    ax = axes[1]
    ax.plot(fno_epochs, fno_val_l2,
            color=COLORS['fno'], linewidth=2.5,
            marker='o', markersize=7,
            label='FNO val L2 error')
    ax.axhline(y=0.05, color='gray', linestyle='--',
               alpha=0.6, linewidth=1, label='5% error threshold')
    ax.fill_between(fno_epochs, fno_val_l2,
                    alpha=0.15, color=COLORS['fno'])
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Relative L2 error')
    ax.set_title('FNO Training — f0=10Hz\n'
                 'Converges to 1.77% error in 200 epochs',
                 color=COLORS['fno'])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.08)
    ax.set_xlim(40, 210)

    plt.suptitle('Training Convergence: PINN vs FNO\n'
                 'PINN fails to converge; FNO learns wave physics',
                 y=1.02, fontsize=12)
    plt.tight_layout()
    path = 'results/figures/fig2_convergence.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()


# ══════════════════════════════════════════════════
# FIGURE 3: Summary bar chart
# ══════════════════════════════════════════════════
def figure3_summary():
    print("Building Figure 3: Summary chart...")

    frequencies = [5, 10, 20, 40, 80]

    fd_times = [21.64, 21.88, 21.78, 21.78, 21.87]

    fno_times = [18.72, 70.66, 5.46, 5.63, 5.77]
    fno_l2    = [0.0091, 0.0179, 1.0, 1.0, 1.0]

    pinn_times = [616100, 604100, 594500, 616500, 823200]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    x = np.arange(len(frequencies))
    w = 0.25

    # ── Top left: Runtime comparison (log scale) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w, fd_times, w, label='FD',
            color=COLORS['fd'], alpha=0.85)
    ax1.bar(x,     fno_times, w, label='FNO',
            color=COLORS['fno'], alpha=0.85)
    ax1.bar(x + w,
            [p/1000 for p in pinn_times], w,
            label='PINN (÷1000)', color=COLORS['pinn'], alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Inference time (ms, log scale)')
    ax1.set_title('Runtime Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frequencies)
    ax1.legend(fontsize=8)

    # ── Top right: FNO L2 error across frequencies ──
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = [COLORS['fno'] if l < 0.1 else COLORS['pinn']
                  for l in fno_l2]
    bars = ax2.bar(x, fno_l2, color=bar_colors, alpha=0.85)
    ax2.axhline(y=0.05, color='gray', linestyle='--',
                alpha=0.7, linewidth=1.5, label='5% threshold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Relative L2 error')
    ax2.set_title('FNO Accuracy vs Frequency\n'
                  '(green=works, red=fails)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frequencies)
    ax2.legend(fontsize=8)
    for bar, l2 in zip(bars, fno_l2):
        label = f'{l2:.3f}' if l2 < 0.1 else 'FAIL'
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 label, ha='center', va='bottom', fontsize=8)

    # ── Bottom left: Slowdown factors ──
    ax3 = fig.add_subplot(gs[1, 0])
    slowdowns = [p/f for p, f in zip(pinn_times, fd_times)]
    ax3.bar(x, slowdowns, color=COLORS['pinn'], alpha=0.85)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Slowdown vs FD (×)')
    ax3.set_title('PINN Slowdown vs FD')
    ax3.set_xticks(x)
    ax3.set_xticklabels(frequencies)
    for i, s in enumerate(slowdowns):
        ax3.text(i, s + 50, f'{s:.0f}×',
                 ha='center', fontsize=8)

    # ── Bottom right: Method comparison summary ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary = [
        ['Method', '5Hz', '10Hz', '20Hz', '40Hz', '80Hz'],
        ['FD', '✓', '✓', '✓', '✓', '✓'],
        ['FNO', '✓ 0.9%', '✓ 1.8%', '✗', '✗', '✗'],
        ['PINN', '✗', '✗', '✗', '✗', '✗'],
    ]

    table = ax4.table(
        cellText=summary[1:],
        colLabels=summary[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(fontweight='bold')
        elif '✓' in cell.get_text().get_text():
            cell.set_facecolor('#C8E6C9')
        elif '✗' in cell.get_text().get_text():
            cell.set_facecolor('#FFCDD2')

    ax4.set_title('Method Reliability Summary',
                  pad=20, fontsize=11)

    plt.suptitle('Seismic Wavefield Simulation Benchmark\n'
                 'FD vs FNO vs PINN across 5–80Hz',
                 fontsize=13, fontweight='bold', y=1.01)

    path = 'results/figures/fig3_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.show()


# ══════════════════════════════════════════════════
# RUN ALL FIGURES
# ══════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating all paper figures...\n")

    print("=" * 50)
    l2 = figure1_wavefield_comparison()
    print(f"Figure 1 complete. FNO L2 error: {l2:.4f}\n")

    print("=" * 50)
    figure2_convergence()
    print("Figure 2 complete.\n")

    print("=" * 50)
    figure3_summary()
    print("Figure 3 complete.\n")

    print("=" * 50)
    print("All figures saved to results/figures/")
    print("fig1_wavefield_comparison.png")
    print("fig2_convergence.png")
    print("fig3_summary.png")
