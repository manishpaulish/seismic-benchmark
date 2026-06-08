"""
PINN Frequency Sweep — 20, 40, 80Hz
=====================================
Completes the PINN side of Table 2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_solver import pinn_solve
import time

# FD baseline for comparison
fd_results = {
    5:  {'runtime': 0.289, 'amp': 1.3967e-06},
    10: {'runtime': 0.333, 'amp': 9.3241e-07},
    20: {'runtime': 0.325, 'amp': 5.1285e-07},
    40: {'runtime': 0.283, 'amp': 1.9018e-07},
    80: {'runtime': 0.283, 'amp': 1.7984e-07},
}

frequencies = [20, 40, 80]
pinn_results = []

print("=" * 70)
print("PINN FREQUENCY SWEEP — 20, 40, 80Hz")
print("Each run ~10 minutes. Total ~30 minutes.")
print("=" * 70)

for f0 in frequencies:
    print(f"\nStarting f0={f0}Hz...")
    t_sweep_start = time.perf_counter()

    try:
        result = pinn_solve(
            f0=f0,
            epochs=5000,
            verbose=True
        )

        fd = fd_results[f0]
        slowdown = result['runtime'] / fd['runtime']
        amp_error = result['max_amp'] / fd['amp']

        pinn_results.append({
            'f0': f0,
            'runtime': result['runtime'],
            'max_amp': result['max_amp'],
            'slowdown': slowdown,
            'amp_error': amp_error,
            'converged': result['max_amp'] < 1e-4
        })

        print(f"\nf0={f0}Hz DONE:")
        print(f"  Runtime: {result['runtime']:.1f}s")
        print(f"  Slowdown: {slowdown:.0f}x")
        print(f"  Amplitude: {result['max_amp']:.4e}")
        print(f"  FD amplitude: {fd['amp']:.4e}")
        print(f"  Amplitude error: {amp_error:.0f}x")

    except Exception as e:
        print(f"  FAILED at f0={f0}Hz: {e}")
        pinn_results.append({
            'f0': f0,
            'runtime': None,
            'error': str(e)
        })

print("\n" + "=" * 70)
print("COMPLETE TABLE — PINN vs FD")
print("=" * 70)
print(f"{'Freq':<8} {'FD(s)':<10} {'PINN(s)':<12} "
      f"{'Slowdown':<12} {'Amp Error':<12}")
print("-" * 70)

all_results = [
    {'f0': 5,  'fd_t': 0.289, 'pinn_t': 616.1,  'slow': 2132, 'aerr': 26000},
    {'f0': 10, 'fd_t': 0.333, 'pinn_t': 604.1,  'slow': 1814, 'aerr': 55000},
]

for r in pinn_results:
    if r.get('runtime'):
        all_results.append({
            'f0': r['f0'],
            'fd_t': fd_results[r['f0']]['runtime'],
            'pinn_t': r['runtime'],
            'slow': r['slowdown'],
            'aerr': r['amp_error']
        })

for r in all_results:
    print(f"{r['f0']:<8} {r['fd_t']:<10.3f} {r['pinn_t']:<12.1f} "
          f"{r['slow']:<12.0f} {r['aerr']:<12.0f}")
