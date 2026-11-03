# Pearl FP16 scheme specification

`fp16_scheme.tex` specifies an alternative Pearl proof-of-useful-work instantiation whose unit
of work is **FP16** matrix multiplication on NVIDIA A100 (GA100, `sm_80`) tensor cores. It
extends the FP8 certificate-v4 specification rather than replacing it: commitments, seed chain,
state window, ticket, target, MoE extension, and the ZK split are shared, and FP8 remains a
separate `Quant` value.

The new idea: hardness comes from the **nonlinearity of the device's accumulation**, not from a
coarse rounding grid. The A100 tensor core truncates each product onto a per-group alignment
grid *before* summing (groups of 8, 24-bit window, round-toward-zero per group). That per-product
truncation does not commute with the reduction, so it cannot be expressed as a matrix
multiplication — the same obstruction that makes the integer transcript scheme hard, applied for
free on every MAC. This lets recovered products carry FP16-level accuracy instead of
FP8-residual accuracy.

The experiments the spec's appendices summarize are reproduced by the committed harness in
[`validation/`](validation/) (A100 HMMA accumulation-model capture + model-vs-silicon cross-check,
RZ-vs-RNE resolution, policy `f_bp`/`rho` calibration, and the truncation-attack cost benchmark).
See [`validation/README.md`](validation/README.md) for how to run it on `sm_80` silicon and the
recorded results.

## Building the PDF

The checked-in `fp16_scheme.pdf` is the authoritative rendering. To rebuild:

```sh
# Any LaTeX engine works; the doc uses only amsmath/amssymb/booktabs/hyperref.
tectonic fp16_scheme.tex        # self-contained, recommended
# or
latexmk -pdf fp16_scheme.tex
```
