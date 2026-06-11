# Seismic Wavefield Simulation Benchmark

**When Do Neural Approaches Outperform Finite Differences for Seismic Wavefield Simulation?**

A systematic benchmark comparing PINNs, FNOs, and classical FD solvers for the 2D acoustic wave equation across the full exploration seismology frequency range of 5 to 80Hz.

**Author:** Manish Paul, IIT Kharagpur
**Paper:** arXiv submission pending
**Code:** All code to reproduce the experiments is in this repository.

---

## Key Findings

1. **FD is fast, accurate, and frequency-independent** - solves all frequencies in under 25ms with machine precision regardless of velocity model complexity.

2. 2. **PINNs fail at every frequency** - spectral bias causes the IC loss to stagnate at 0.76, producing slowdowns of 1,814 to 2,909x and amplitude errors exceeding 26,000x relative to FD.
  
3. 3. **FNO works at low frequencies** - achieves 0.91% L2 error at 5Hz and 1.79% at 10Hz on homogeneous media.

4. 4. **FNO fails above 10Hz due to amplitude degeneracy** - a previously undocumented failure mode where near-uniform wavefield amplitudes across training samples flatten the MSE loss surface.
        
5. 5. **FNO degrades on layered media** - L2 error increases 36-fold at 5Hz and fails completely at 10Hz on a 4-layer model, even with model-specific retraining.
           
6. ---
           
7. ## Results Summary
           
8. | Method | 5Hz | 10Hz | 20Hz | 40Hz | 80Hz |
9. |--------|-----|------|------|------|------|
10. | FD | under 25ms | under 25ms | under 25ms | under 25ms | under 25ms |
11. | FNO (homogeneous) | 0.91% L2 | 1.79% L2 | FAIL | FAIL | FAIL |
12. | FNO (layered) | 32.4% L2 | FAIL | - | - | - |
13. | PINN | 2132x slow | 1814x slow | 1829x slow | 2178x slow | 2909x slow |
           
14. ---
           
15. ## Setup
           
16. ```bash
pip install torch numpy matplotlib deepxde neuraloperator
```
All experiments run on Apple M1 MacBook Air (8GB RAM) with MPS acceleration.

---

## Reproducing Results

```bash
python3 fd_solver/frequency_benchmark.py      # FD baseline
python3 pinn_solver/pinn_frequency_sweep.py   # PINN benchmark
python3 fno_solver/fno_v2.py                  # FNO homogeneous
python3 fno_solver/fno_layered.py             # FNO layered
python3 notebooks/visualisation.py            # Paper figures
```

---

## Paper

**Title:** When Do Neural Approaches Outperform Finite Differences for Seismic Wavefield Simulation? A Systematic Comparison of PINNs, Fourier Neural Operators, and Classical Solvers

The full paper is in `paper/main.tex`. All 8 citations verified. Compiled PDF available on request.

---

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

---

## License

MIT License.
