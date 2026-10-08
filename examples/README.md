# Example dataset

A ~57 MB subset of the real training data, provided so that every command in the
top-level README runs out of the box:

```
examples/
├── real/
│   ├── 2_Varex-Wenqian-Nickel-Ni83_ch3/   1024x1024 crop, all three classes
│   └── 7_GE-pliaw_nov25-CoNiV_Sam4/       1024x1024 crop, all three classes
├── synth_train.h5          48 frames   old synthetic (diverse materials)
├── synth_val.h5            16 frames
├── filtered_indices.json   index lists for the two files above
├── ring_train.h5           48 patches  ring synthetic (powder-rich)
└── ring_val.h5             16 patches
```

The schema is identical to the full dataset, so `--data-root examples` exercises
the complete pipeline: 3-way mixing, weighted sampling, the three validation
domains and the combined metric.

**This subset cannot reproduce the published results.** The full training set is
roughly 200x larger (7 full-resolution hand-labelled frames tiled into 314
patches, 6,412 old-synthetic frames, 10,000 ring patches). Use it for smoke
tests, for reading the data path in a debugger, and for checking that a
checkpoint loads — not for training or for reporting numbers.

Rebuild it from the full dataset with `scripts/make_example_data.py`.
