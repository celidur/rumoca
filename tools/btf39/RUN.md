# be_harness_btf : lancer sur le 39 bus

À exécuter dans `/home/kerman/rumoca-gpu`, en CPU local (pas de GPU, pas de Slurm).
Copier `be_harness_btf.cpp` à côté de `be_harness.cpp`.

```sh
M=MSEMT_Examples_IEEE39Bus_IEEE39BusCaseIdealSRC_CP_IEEE39Bus
U=MSEMT_EXAMPLES_IEEE39BUS_IEEE39BUSCASEIDEALSRC_CP_IEEE39BUS
D=benchmark-runs/btf39; mkdir -p $D
for B in 1 0; do
  g++ -std=c++17 -O1 -w -DTHETA=1.0 -DMAXDEPTH=12 -DBTF=$B \
    -DMODEL_SRC="\"IEEE39Bus_cuda/${M}_ode.cu\"" -DM=$M \
    -DY_LEN=${U}_Y_LEN -DP_LEN=${U}_P_LEN -DNSTATE=${U}_STATE_COUNT \
    -DHSTRIDE=${U}_DELAY_HISTORY_STRIDE be_harness_btf.cpp -o $D/be_btf$B
done
sha256sum be_harness_btf.cpp IEEE39Bus_cuda/${M}_ode.cu | tee $D/SHA256SUMS
# 1) court essai (2 ms) pour comparer les deux modes
( cd $D && ./be_btf1 0.002 25e-6 btf1_2ms.csv | tee btf1_2ms.log )
( cd $D && ./be_btf0 0.002 25e-6 btf0_2ms.csv | tee btf0_2ms.log )
# 2) run complet BTF
( cd $D && /usr/bin/time -v ./be_btf1 0.1 25e-6 btf1_100ms.csv > btf1_100ms.log 2>&1 )
```

À rapporter :
- les lignes `structure #k` et `pattern grew by` : est-ce que le motif bouge après t=0 ?
- `unconverged`, `halvings`, `rhs_calls` (dont la part Jacobien) et `wall` contre 33 min ;
- la ligne `time:` (rhs, jacobian, probes, lu, solve) ;
- l'écart des 12 premiers états contre `states_25us_first12.csv` (une ligne par ms) : `python3 -I compare.py states_25us_first12.csv btf1_100ms.csv`.

Options : `-DREUSE_JF=1` réutilise le Jacobien quand h change (par défaut 0, la règle d'origine, 13 % plus rapide sur le 39 bus).
`-DPROBE_EVERY=N` (400 par défaut, soit 10 ms à 25 µs) espace les sondes denses.
`-DBTF=0 -DREUSE_JF=0` redonne exactement `be_harness.cpp`, vérifié bit à bit sur le modèle `mock_ode.cu`.
