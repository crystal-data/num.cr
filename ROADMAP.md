### This library seeks to provide a public API similar to NumPy, this list keeps track of implemented routines.

This is a great place to start looking for ways to contribute.  All of these
functions would be awesome to have, so if you can implement any or many of them,
go for it!

### Array Creation Routines

Ones and Zeros

- [x] `empty`
- [x] `empty_like`
- [x] `eye`
- [x] `identity`
- [x] `ones`
- [x] `ones_like`
- [x] `zeros`
- [x] `zeros_like`
- [x] `full`
- [x] `full_like`

From Existing Data

- [ ] `frombuffer`
- [ ] `fromfile`
- [ ] `from_function` (Should probably be from proc or something similar)
- [x] `from_iter` (Implemented by the `new` from block routines)
- [ ] `fromstring`
- [ ] `loadtxt`

Numerical Ranges

- [x] `arange`
- [x] `linspace`
- [x] `logspace`
- [x] `geomspace`
- [ ] `meshgrid`
- [x] `mgrid`
- [x] `ogrid`

Building Matrices

- [x] `diag`
- [ ] `diagflat`
- [x] `tri`
- [x] `tril`
- [x] `triu`
- [x] `vander`

### Array Manipulation Routines

Basic Operations

- [ ] `copyto`

Changing Array Shape

- [x] `reshape`
- [x] `ravel` (Still needs to be added as a standalone method)
- [x] `flat`
- [x] `flatten`

Tranpose-like Operations

- [x] `transpose`

Changing Number of Dimensions

- [x] `atleast_1d`
- [x] `atleast_2d`

Joining Tensors and Matrices

- [x] `concatenate`
- [ ] `stack`
- [x] `column_stack`
- [x] `dstack`
- [x] `hstack`
- [x] `vstack`

Splitting Tensors and Matrices

- [ ] `split`
- [ ] `hsplit`
- [ ] `vsplit`

Tilings Tensors and Matrices

- [ ] `tile`
- [ ] `repeat`

Adding and Removing Elements

- [ ] `delete`
- [ ] `insert`
- [ ] `append`
- [ ] `resize`
- [ ] `trim_zeros`
- [ ] `unique`

Rearranging Elements

- [ ] `flip`
- [ ] `fliplr`
- [ ] `flipud`
- [x] `reshape`
- [ ] `roll`
- [ ] `rot90`

### Binary Operations

Bit Operations

- [x] `bitwise_and`
- [x] `bitwise_or`
- [x] `bitwise_xor`
- [ ] `invert`
- [x] `left_shift`
- [x] `right_shift`

### Indexing Operations

Generating Index Tensors and Matrices

- [ ] `nonzero`
- [ ] `where`
- [ ] `indices`
- [ ] `diag_indices`
- [ ] `diag_indices_from`
- [ ] `mask_indices`
- [ ] `tril_indices`
- [ ] `tril_indices_from`
- [ ] `triu_indices`
- [ ] `triu_indices_from`

Indexing Like Operations

- [ ] `take`
- [ ] `take_along_axis`
- [ ] `choose`
- [ ] `compress`
- [x] `diag`
- [x] `diagonal`
- [ ] `select`

Inserting data into Tensors and Matrices

- [ ] `place`
- [ ] `put`
- [ ] `put_along_axis`
- [ ] `putmask`
- [ ] `fill_diagonal`

### Linear Algebra

Matrix and Vector Products

- [x] `dot`
- [ ] `multi_dot`
- [ ] `inner`
- [x] `outer`
- [x] `matmul`
- [ ] `einsum` (This is a monster task, would love help)
- [ ] `matrix_power`
- [ ] `kron`

Decompositions

- [ ] `cholesky`
- [ ] `qr`
- [ ] `svd`

Matrix Eigenvalues

- [ ] `eig`
- [ ] `eigh`
- [ ] `eigvals`
- [ ] `eigvalsh`

Norms and the like

- [x] `norm`
- [ ] `cond`
- [ ] `det`
- [ ] `matrix_rank`
- [ ] `slogdet`
- [x] `trace`

Solving equations and inverting matrices

- [ ] `solve`
- [ ] `tensorsolve`
- [ ] `lstsq`
- [x] `inv`
- [ ] `pinv`
- [ ] `tensorinv`

### Mathematical Functions

Trigonometric Functions

- [x] `sin`
- [x] `cos`
- [x] `tan`
- [x] `asin`
- [x] `acos`
- [x] `atan`
- [x] `hypot`
- [x] `atan2`
- [x] `degrees`
- [x] `radians`
- [ ] `unwrap`

Hyperbolic Functions

- [x] `sinh`
- [x] `cosh`
- [x] `tanh`
- [x] `asinh`
- [x] `acosh`
- [x] `atanh`

Rounding

- [ ] `around`
- [ ] `round`
- [ ] `rint`
- [ ] `fix`
- [ ] `floor`
- [ ] `ceil`
- [ ] `trunc`

Sums, product and differences

- [x] `prod`
- [x] `sum`
- [x] `cumprod`
- [x] `cumsum`
- [ ] `diff`
- [ ] `ediff1d`
- [ ] `gradient`
- [ ] `cross`
- [ ] `trapz`

Exponents and Logarithms

- [x] `exp`
- [x] `expm1`
- [x] `exp2`
- [x] `log`
- [x] `log10`
- [x] `log2`
- [x] `log1p`
- [ ] `logaddexp`
- [ ] `logaddexp2`

Special Functions

- [x] `i0`
- [ ] `sinc`

Floating Point Routines

- [ ] `signbit`
- [x] `copysign`
- [x] `frexp`
- [x] `ldexp`
- [ ] `nextafter`
- [ ] `spacing`

Rational Routines

- [ ] `lcm`
- [ ] `gcd`

Arithmetic Operations

- [x] `add`
- [ ] `reciprocal`
- [ ] `positive`
- [ ] `negative`
- [x] `multiply`
- [x] `divide`
- [x] `power`
- [x] `subtract`
- [ ] `true_divide`
- [x] `floor_divide`
- [x] `mod`
- [ ] `modf`
- [x] `remainder`
- [ ] `divmod`

Miscellaneous

- [ ] `convolve`
- [ ] `clip`
- [x] `sqrt`
- [x] `cbrt`
- [ ] `square`
- [ ] `absolute`
- [ ] `fabs`
- [ ] `sign`
- [ ] `heaviside`
- [x] `maximum`
- [x] `minimum`
- [ ] `fmax`
- [ ] `fmin`
- [ ] `nan_to_num`
- [ ] `real_if_close`
- [ ] `interp`

### Sorting, Searching and Counting

Sorting

- [ ] `sort`
- [ ] `lexsort`
- [ ] `argsort`
- [ ] `msort`
- [ ] `partition`
- [ ] `argpartition`

Searching

- [x] `argmax`
- [ ] `nanargmax`
- [x] `argmin`
- [ ] `nanargmin`
- [ ] `argwhere`
- [ ] `nonzero`
- [ ] `flatnonzero`
- [ ] `where`
- [ ] `searchsorted`
- [ ] `extract`

Counting

- [ ] `count_nonzero`
- [x] `bincount`
