![num.cr](https://raw.githubusercontent.com/crystal-data/bottle/rename/static/numcr_logo.png)

[![Join the chat at https://gitter.im/crystal-data/bottle](https://badges.gitter.im/crystal-data/bottle.svg)](https://gitter.im/crystal-data/bottle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![Crystal CI](https://github.com/crystal-data/num.cr/workflows/Crystal%20CI/badge.svg)

Num.cr is the core shard needed for scientific computing with Crystal

- **Website:** https://crystal-data.github.io/num.cr
- **API Documentation:** https://crystal-data.github.io/num.cr/
- **Source code:** https://github.com/crystal-data/num.cr
- **Bug reports:** https://github.com/crystal-data/num.cr/issues

It provides:

- An n-dimensional `Tensor` data structure
- Efficient `map`, `reduce` and `accumulate` routines
- GPU accelerated routines backed by `OpenCL`
- Linear algebra routines backed by `LAPACK` and `BLAS`

## Prerequisites

`Num.cr` aims to be a scientific computing library written in pure Crystal.
All standard operations and data structures are written in Crystal.  Certain
routines, primarily linear algebra routines, are instead provided by a
`BLAS` or `LAPACK` implementation.

Several implementations can be used, including `Cblas`, `Openblas`, and the
`Accelerate` framework on Darwin systems.  For GPU accelerated `BLAS` routines,
the `ClBlast` library is required.

`Num.cr` also supports `Tensor`s stored on a `GPU`.  This is currently limited
to `OpenCL`, and a valid `OpenCL` installation and device(s) are required.

## Installation

Add this to your applications `shard.yml`

```
dependencies:
  num:
    github: crystal-data/num.cr
```

Several third-party libraries are required to use certain features of `Num.cr`.
They are:

- BLAS
- LAPACK
- OpenCL
- ClBlast
- NNPACK

While not at all required, they provide additional functionality than is
provided by the basic library.

## Just show me the code

The core data structure implemented by `Num.cr` is the `Tensor`, an N-dimensional
data structure.  A `Tensor` supports slicing, mutation, permutation, reduction,
and accumulation.  A `Tensor` can be a view of another `Tensor`, and can support
either C-style or Fortran-style storage.

### Creation

There are many ways to initialize a `Tensor`.  Most creation methods can
allocate a `Tensor` backed by either `CPU` or `GPU` based storage.

```crystal
[1, 2, 3].to_tensor
Tensor.from_array [1, 2, 3]
Tensor(UInt8, CPU(UInt8)).zeros([3, 3, 2])
Tensor.random(0.0...1.0, [2, 2, 2])

Tensor(Float32, OCL(Float32)).zeros([3, 2, 2])
Tensor(Float64, OCL(Float64)).full([3, 4, 5], 3.8)
```

### Operations

A `Tensor` supports a wide variety of numerical operations.  Many of these
operations are provided by `Num.cr`, but any operation can be mapped across
one or more `Tensor`s using sophisticated broadcasted mapping routines.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor

puts a + b

# a is broadcast to b's shape
# [[ 4,  6,  8, 10],
#  [ 6,  8, 10, 12]]
```

When operating on more than two `Tensor`s, it is recommended to use `map`
rather than builtin functions to avoid the allocation of intermediate
results.  All `map` operations support broadcasting.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor
c = [3, 5, 7, 9].to_tensor

a.map(b, c) do |i, j, k|
  i + 2 / j + k * 3.5
end

# [[12.1667, 20     , 27.9   , 35.8333],
#  [11.9   , 19.8333, 27.7857, 35.75  ]]
```

### Mutation

`Tensor`s support flexible slicing and mutation operations.  Many of these
operations return views, not copies, so any changes made to the results might
also be reflected in the parent.

```crystal
a = Tensor.new([3, 2, 2]) { |i| i }

puts a.transpose

# [[[ 0,  4,  8],
#   [ 2,  6, 10]],
#
#  [[ 1,  5,  9],
#   [ 3,  7, 11]]]

puts a.reshape(6, 2)

# [[ 0,  1],
#  [ 2,  3],
#  [ 4,  5],
#  [ 6,  7],
#  [ 8,  9],
#  [10, 11]]

puts a[..., 1]

# [[ 2,  3],
#  [ 6,  7],
#  [10, 11]]

puts a[1..., {..., -1}]

# [[[ 6,  7],
#   [ 4,  5]],
#
#  [[10, 11],
#   [ 8,  9]]]

puts a[0, 1, 1].value

# 3
```

### Linear Algebra

`Tensor`s provide easy access to power Linear Algebra routines backed by
LAPACK and BLAS implementations, and ClBlast for GPU backed `Tensor`s.

```crystal
a = [[1, 2], [3, 4]].to_tensor.map &.to_f32

puts a.inv

# [[-2  , 1   ],
#  [1.5 , -0.5]]

puts a.eigvals

# [-0.372281, 5.37228  ]

puts a.matmul(a)

# [[7 , 10],
#  [15, 22]]
```

### Einstein Notation

For representing certain complex contractions of `Tensor`s, Einstein notation
can be used to simplify the operation.  For example, the following matrix
multiplication + summation operation:

```
a = Tensor.new([30, 40, 50]) { |i| i * 1_f32 }
b = Tensor.new([40, 30, 20]) { |i| i * 1_f32 }

result = Float32Tensor.zeros([50, 20])
ny, nx = result.shape
b2 = b.swap_axes(0, 1)
ny.times do |k|
  nx.times do |l|
    result[k, l] = (a[..., ..., k] * b2[..., ..., l]).sum
  end
end
```

Can instead be represented in Einstein notiation as the following:

```
Num::Einsum.einsum("ijk,jil->kl", a, b)
```

This can lead to performance improvements due to optimized contractions
on `Tensor`s.

```
einsum   2.22k   (450.41µs) (± 0.86%)   350kB/op        fastest
manual   117.52  (  8.51ms) (± 0.98%)  5.66MB/op  18.89× slower
```

### Machine Learning

`Num::Grad` provides a pure-crystal approach to find derivatives of
mathematical functions.  Use a `Num::Grad::Variable` with a `Num::Grad::Context`
to easily compute these derivatives.

```crystal
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

x = ctx.variable([3.0].to_tensor)
y = ctx.variable([2.0].to_tensor)

# f(x) = x ** y
f = x ** y
puts f # => [9]

f.backprop

# df/dx = y * x = 6.0
puts x.grad # => [6.0]
```

`Num::NN` contains an extension to `Num::Grad` that provides an easy-to-use
interface to assist in creating neural networks.  Designing and creating
a network is simple using Crystal's block syntax.

```crystal
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

x_train = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].to_tensor
y_train = [[0.0], [1.0], [1.0], [0.0]].to_tensor

x = ctx.variable(x_train)

net = Num::NN::Network.new(ctx) do
  input [2]
  # A basic network with a single hidden layer using
  # a ReLU activation function
  linear 3
  relu
  linear 1

  # SGD Optimizer
  sgd 0.7

  # Sigmoid Cross Entropy to calculate loss
  sigmoid_cross_entropy_loss
end

500.times do |epoch|
  y_pred = net.forward(x)
  loss = net.loss(y_pred, y_train)
  puts "Epoch: #{epoch} - Loss #{loss}"
  loss.backprop
  net.optimizer.update
end

# Clip results to make a prediction
puts net.forward(x).value.map { |el| el > 0 ? 1 : 0}

# [[0],
#  [1],
#  [1],
#  [0]]
```

Review the documentation for full implementation details, and if something is missing,
open an issue to add it!
