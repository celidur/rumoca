import json, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
d=json.load(open('jacobian.json')); A=np.array([[np.nan if v is None else v for v in r] for r in d['state_jacobian']['matrix']]); n=len(A)
print('nan',np.isnan(A).sum()); A=np.nan_to_num(A)
P=(A!=0)|np.eye(n,dtype=bool)
ns,lab=connected_components(sp.csr_matrix(P),directed=True,connection='strong')
nw,labw=connected_components(sp.csr_matrix(P),directed=False)
print('strong',ns,'weak',nw)
off=sum(1 for i,j in zip(*np.nonzero(A)) if lab[i]!=lab[j]); print('nnz outside SCC diag blocks',off)
# weak comps sizes
from collections import Counter; print('weak sizes',sorted(Counter(labw).values(),reverse=True))
for h in (25e-6,):
  M=np.eye(n)-h*A
  ev=np.linalg.eigvals(A); print('eig J: max|re|',np.max(np.abs(ev.real)),'max re',ev.real.max(),'min re',ev.real.min(), 'max |im|',np.abs(ev.imag).max())
  print('h*max|lambda|',h*np.abs(ev).max())
  print('cond(I-hJ)',np.linalg.cond(M))
  # per block cond
  cs=[]
  for c in range(ns):
    idx=np.where(lab==c)[0]; cs.append((len(idx),np.linalg.cond(M[np.ix_(idx,idx)])))
  print('worst block cond',max(cs,key=lambda x:x[1]))
# flop estimate
sizes=np.bincount(lab); print('dense LU flops',2*n**3/3,'block LU flops',sum(2*s**3/3 for s in sizes), 'ratio',(2*n**3/3)/sum(2*s**3/3 for s in sizes))
print('dense solve flops',2*n*n,'block',sum(2*s*s for s in sizes)+2*off)
