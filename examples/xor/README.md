# XOR example

A 2-layer MLP (`X[B,2] -> dense -> tanh -> dense -> logits[B,1] -> BCE-with-logits`) trained with full-batch Adam, built only from the core training ops (dense forward/backward, tanh, BCE-with-logits, Adam); also wired as a fast CPU unit test.

Init runs on host mirrors with a seeded mt19937 (Xavier uniform weights, zero biases) so runs are reproducible. Scratch and gradient buffers are allocated once per batch/hidden width and reused across epochs (no per-epoch allocation in the loop).

Layout:

- `KokkosDNN_XorMLP.hpp` — header-only MLP (`Dataset`, `MLP` + Adam moments, `Workspace`, `create_mlp`, `forward`, `train_epoch`, `predict_probs`).
- `xor_train.cpp` — `xor_train` training driver (prints the final truth table).
- `Test_Xor.cpp` — gtest (gradient sanity checks plus an end-to-end train-to-truth-table test).

## Run

```bash
./build/examples/xor/xor_train [--hidden N] [--epochs N] [--lr X] [--seed N] [--print-every N]
```

Defaults: `--hidden 8 --epochs 5000 --lr 0.05 --seed 0 --print-every 500`.
The driver prints per-epoch loss, then the truth-table probabilities and the
final `correct n/4` line (exit code is 0 only when all four cases classify
correctly). The `Test_Xor.TrainsToTruthTable` gtest trains the same
`2->8(tanh)->1` MLP with full-batch Adam on `DefaultExecutionSpace` and
expects loss `< 0.05` with every case within `0.2` of its label.
