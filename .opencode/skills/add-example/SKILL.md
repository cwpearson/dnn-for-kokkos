---
name: Add Example
description: Add a new example under examples/ following repo conventions
---

# Add an Example

Use this workflow when adding a new example under `examples/<name>/`.
Examples are NOT meant to be comprehensive, feature-rich binaries with a combinatorial explosion of capabilities; rather, targeted demonstrations of the core library.
They are never part of `libkokkosdnn`.

## README.md

Every example has a README with these sections (see `examples/xor/README.md`
for the minimal form, `examples/llama/README.md` for the full form):

```markdown
# <Name> example
<One paragraph: what the example does. Scope of the example.>

## Quickstart
<cmake -S . -B build -DKokkos_ROOT=... ; cmake --build; hf download if needed; driver run>

## Example commands
<tiny no-download run first, then real-checkpoint, then flag variants>

## Notes
<Layout: file -> responsibility; arch dims; tensor names; tokenizer/chat/BOS/stop; KV/prefill>

## Verification
ctest --test-dir build -R <Name> --output-on-failure

## Performance
<shapes, memory footprint, TPL dispatch notes, what not to benchmark on Serial>
```

