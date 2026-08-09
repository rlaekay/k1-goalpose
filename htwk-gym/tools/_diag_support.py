"""착지 검출이 왜 실패했는지 — support 열 분포를 본다."""
import glob
import numpy as np

for p in sorted(glob.glob("logs/mujoco/mediator/*_series.npz")):
    z = np.load(p, allow_pickle=True)
    s, c = z["state"], [str(x) for x in z["state_cols"]]
    if s.size == 0:
        print(p, "state 비어 있음")
        continue
    sup = s[:, c.index("support")]
    u, n = np.unique(sup, return_counts=True)
    tr = int(((sup[1:] >= 2) & (sup[:-1] < 2)).sum())
    tr1 = int(((sup[1:] == 2) & (sup[:-1] == 1)).sum())
    print("%-28s rows %5d  support %s  0->2/1->2 전이 %d (1->2 %d)  낙상 %d"
          % (p.split("/")[-1], len(s), dict(zip(u.astype(int), n)), tr, tr1,
             len(z["fall_t"])))
