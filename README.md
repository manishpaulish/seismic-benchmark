# Seismic Wavefield Simulation Benchmark

**When Do Neural Approaches Outperform Finite Differences for Seismic Wavefield Simulation?**

A systematic benchmark comparing PINNs, FNOs, and classical FD solvers for the 2D acoustic wave equation across 5 to 80Hz.

**Author:** Manish Paul, IIT Kharagpur | **Paper:** arXiv pending | **Code:** fully reproducible

## Key Findings

**FD is fast, accurate, and frequency-independent** - solves all frequencies in under 25ms with machine precision regardless of velocity model complexity.

**PINNs fail at every frequency** - spectral bias causes IC loss to stagnate at 0.76, producing slowdowns of 1814x to 2909x and amplitude errors exceeding 26000x relative to FD.

**FNO works at low frequencies** - achieves 0.91% L2 error at 5Hz and 1.79% at 10Hz on homogeneous media.

**FNO fails above 10Hz due to amplitude degeneracy** - a previously undocumented failure mode where near-uniform wavefield amplitudes flatten the MSE loss surface.

**FNO degrades on layered media** - L2 error increases 36-fold at 5Hz and fails completely at 10Hz, even with model-specific retraining.

## Results Summary

| Method | 5Hz | 10Hz | 20Hz | 40Hz | 80Hz |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FD | under 25ms | under 25ms | under 25ms | under 25ms | under 25ms |
| FNO homogeneous | 0.91% L2 | 1.79% L2 | FAIL | FAIL | FAIL |
| FNO layered | 32.4% L2 | FAIL | - | - | - |
| PINN | 2132x | 1814x | 1829x | 2178x | 2909x |

## Setup

```bash
pip install torch numpy matplotlib deepxde neuraloperator
```

All experiments run on Apple M1 MacBook Air 8GB RAM with MPS acceleration.

## Reproducing Results

```bash
python3 fd_solver/frequency_benchmark.py
python3 pinn_solver/pinn_frequency_sweep.py
python3 fno_solver/fno_v2.py
python3 fno_solver/fno_layered.py
python3 notebooks/visualisation.py
```

## Paper

**Title:** When Do Neural Approaches Outperform Finite Differences for Seismic Wavefield Simulation? A Systematic Comparison of PINNs, Fourier Neural Operators, and Classical Solvers

Full paper in `paper/main.tex`. All 8 citations verified. PDF available on request.

## Citation

```bibtex
@misc{paul2026seismic,
  title={When Do Neural Approaches Outperform Finite Differences for Seismic Wavefield Simulation?},
  author={Paul, Manish},
  year={2026},
  institution={Indian Institute of Technology Kharagpur},
  note={arXiv preprint, identifier pending}
}
```

## License

MIT License.
