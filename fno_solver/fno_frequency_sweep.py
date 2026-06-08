"""
FNO Frequency Sweep — 5, 20, 40, 80Hz
Completes the three-way comparison table.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_v2 import benchmark_fno_v2

frequencies = [5, 20, 40, 80]

# Results from previous runs
results = {
    10: {'train_time': 824.3, 'avg_inference_ms': 70.66,
         'avg_fd_ms': 21.88, 'avg_l2': 0.0179}
}

print("=" * 60)
print("FNO FREQUENCY SWEEP — 5, 20, 40, 80Hz")
print("Estimated time: ~60 minutes total")
print("=" * 60)

for f0 in frequencies:
    print(f"\nStarting f0={f0}Hz...")
    try:
        result = benchmark_fno_v2(f0=f0)
        results[f0] = result
        print(f"f0={f0}Hz complete: L2={result['avg_l2']:.4f}, "
              f"inference={result['avg_inference_ms']:.2f}ms")
    except Exception as e:
        print(f"FAILED at f0={f0}Hz: {e}")

print("\n" + "=" * 60)
print("COMPLETE THREE-WAY TABLE")
print("=" * 60)
print(f"{'Freq':<8} {'FD(ms)':<10} {'FNO(ms)':<12} "
      f"{'FNO L2':<12} {'PINN(ms)':<14} {'PINN L2':<10}")
print("-" * 60)

pinn_data = {
    5:  {'ms': 616100, 'l2': 74.0},
    10: {'ms': 604100, 'l2': 74.0},
    20: {'ms': 594500, 'l2': 74.0},
    40: {'ms': 616500, 'l2': 74.0},
    80: {'ms': 823200, 'l2': 74.0},
}

for f0 in [5, 10, 20, 40, 80]:
    if f0 in results:
        r = results[f0]
        p = pinn_data[f0]
        fd_ms = r.get('avg_fd_ms', r.get('avg_fd_ms', 0))
        fno_ms = r.get('avg_inference_ms', 0)
        fno_l2 = r.get('avg_l2', 0)
        print(f"{f0:<8} {fd_ms:<10.2f} {fno_ms:<12.2f} "
              f"{fno_l2:<12.4f} {p['ms']:<14} {p['l2']:<10}")
