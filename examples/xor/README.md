# XOR example

Trains a 2-layer MLP (`2 -> hidden(tanh) -> 1`, BCE-with-logits, full-batch Adam) to learn the XOR truth table, using only core training ops (dense forward/backward, tanh, BCE-with-logits, Adam).

## Quickstart (0 to working)

No downloads needed — the dataset is the four XOR pairs generated in code, and there are no Hugging Face weights.

```bash
# 1. Build and install Kokkos once (Serial is enough)
git clone --branch 5.2.1 --recursive https://github.com/kokkos/kokkos.git kokkos-src
cmake -S kokkos-src -B kokkos-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/kokkos-install \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build kokkos-build --parallel 8
cmake --install kokkos-build

# 2. Configure and build the example target
cmake -S . -B build \
  -DKokkos_ROOT=/path/to/kokkos-install \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8 --target xor_train
./build/examples/xor/xor_train
```

## Example commands

```bash
./build/examples/xor/xor_train [--hidden N] [--epochs N] [--lr X] [--seed N] [--print-every N]
./build/examples/xor/xor_train --hidden 16 --epochs 2000 --lr 0.05 --print-every 500
./build/examples/xor/xor_train --hidden 8 --epochs 5000 --lr 0.05 --seed 1 --print-every 0
ctest --test-dir build -R Xor --output-on-failure
```

Defaults: `--hidden 8 --epochs 5000 --lr 0.05 --seed 0 --print-every 500`. `--print-every 0` disables per-epoch logging. Exit code is 0 only when all four cases classify correctly.

## Notes

- Layout: `KokkosDNN_XorMLP.hpp` (header-only `Dataset`, `MLP` + Adam moments, `Workspace`, `create_mlp`, `forward`, `train_epoch`, `predict_probs`), `xor_train.cpp` (driver, prints final truth table), `Test_Xor.cpp` (gtest).
- Init runs on host mirrors with a seeded mt19937 (Xavier uniform weights, zero biases) so runs are reproducible. Scratch and gradient buffers are allocated once per batch/hidden width and reused across epochs.

## Verification

- The driver prints per-epoch loss, then the truth-table probabilities and a `final loss ..., correct n/4` line.
- The `Test_Xor.TrainsToTruthTable` gtest trains the same `2->8(tanh)->1` MLP with full-batch Adam on `DefaultExecutionSpace` and expects loss `< 0.05` with every case within `0.2` of its label.

## Performance

Trivial compute (4-sample full batch); it runs in seconds on Serial CPU. Increase `--hidden`/`--epochs` only to experiment with convergence, not for benchmarking.
