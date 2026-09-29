# Parent selection: simulation results (2026-09-28)

Code under test: `kernel/ar_kernel/selection.py` as shipped (softmax over subtree value with a size
penalty; `noise_floor` 4.4e-4, temperature 2.0, epsilon 0.2). Pinned by `tests/test_selection.py`.

| Situation (6-node chain) | P(newest) | P(node with true +0.02) |
|---|---|---|
| pure noise, score SD 0.008 (500 chains) | 0.206 (uniform: 0.167) | n/a |
| one node truly +0.02, others equal | n/a | 0.35 (0.30 with `noise_floor` 0.008) |

The acceptance run (root 0.7871, n1 0.7945, n2 0.7917, n3 0.7907): at the last draw the newest node
held P = 0.25 and the best node 0.37, so picking the newest each time was luck (joint probability
about 0.09), not a bug. The newest leaf gets size penalty 1 by construction, which gives the mild
recency bias in the first row. No change to `selection.py`; `noise_floor` stays as configured.
