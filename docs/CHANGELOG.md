# Changes since the submitted version

**IEMOCAP.** The submitted results had two errors. The MoMKE fold dumps store logits
under `probs`, and these were read as probabilities. Each mask dump also orders its
retrieval pool differently, so richer-view outputs were paired with the wrong pool
rows. The drivers now apply `guard.losses.softmax` and align rows through
`experiments/iemocap_io.py`, and `guard.losses.check_outputs` refuses any output that
is not a probability. Frozen accuracies do not change.

**Audit.** `experiments/rebuttal_audit.py` checks the inputs of all seven benchmarks
directly and re-creates the old IEMOCAP pairing as a control.

**New studies.** `docs/PAPER_MAP.md` lists their drivers and outputs.
