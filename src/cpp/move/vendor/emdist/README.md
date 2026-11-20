emdist 0.3-3 (CRAN, 2023-06-13), the C code of the exact Earth Mover's Distance: Yossi Rubner's emd.c (1998) as
Simon Urbanek adapted it for R. MIT licence (LICENSE). `emd_rubner.cpp` and `emd_rubner.hpp` are `src/emd-rubner.c` and
`src/emd-rubner.h` of the package with the R interface (`Rf_error`, `Rf_warning`, `Rprintf`, the R headers) replaced
by an exception and a flag; the algorithm and its arithmetic (the cost matrix and the weights in `float`) are
unchanged, so that `tf.move.ud_distance(method="exact")` gives the bits of `emdist::emd`.
