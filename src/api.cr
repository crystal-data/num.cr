require "./dtypes/*"

require "./tensor/build"
require "./tensor/creation"
require "./tensor/random"
require "./tensor/linalg"
require "./tensor/operators"
require "./tensor/reductions"
require "./tensor/tensor"
require "./tensor/manipulate"

require "./cl_tensor/cl_tensor"
require "./cl_tensor/creation"
require "./cl_tensor/linalg"

require "./scikit/matrices"
require "./scikit/clustering/kmeans"

require "./libs/local"
require "./libs/nnpack"
require "./libs/plplot"

require "./grad/primitives/*"
require "./grad/extensions/float"
require "./grad/gates_arithmetic"
require "./grad/gates_blas"
require "./grad/gates_trigonometric"
require "./grad/gates_exp"
require "./grad/variable_ops"
require "./grad/infer"
require "./grad/utils"

require "./nn/primitives/*"
require "./nn/initialization"
require "./nn/layers/*"
require "./nn/gates/*"
require "./nn/optimizer"
require "./nn/loss"
require "./nn/network"

require "./nn/datasets/*"

require "./plot/internal/*"
require "./plot/figures/*"
require "./plot/plot"

require "./sparse/csr"
require "./sparse/csc"
require "./sparse/coo"
require "./sparse/operators"

require "./stats/summary"

require "./io/image"
