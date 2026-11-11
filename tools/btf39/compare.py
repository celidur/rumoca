# compare.py REF.csv RUN.csv : écart des 12 premiers états aux instants de REF.
import sys
import numpy as np
ref = np.loadtxt(sys.argv[1], delimiter=',', ndmin=2)
run = np.loadtxt(sys.argv[2], delimiter=',', ndmin=2)
idx = [int(np.argmin(np.abs(run[:, 0] - t))) for t in ref[:, 0]]
keep = [k for k, i in enumerate(idx) if abs(run[i, 0] - ref[k, 0]) < 1e-9]
r, x = ref[keep, 1:13], run[[idx[k] for k in keep], 1:13]
peak = np.maximum(np.abs(r).max(axis=0), 1e-300)
err = np.abs(x - r).max(axis=0)
print(f"instants comparés: {len(keep)}/{len(ref)} (t <= {ref[keep[-1],0] if keep else 0})")
for j in range(r.shape[1]):
    print(f"y[{j:2d}] max|err|={err[j]:.3e}  pic={peak[j]:.3e}  {100*err[j]/peak[j]:.4f} % du pic")
print(f"pire: {100*(err/peak).max():.4f} % du pic")
