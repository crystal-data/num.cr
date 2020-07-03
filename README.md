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
Tensor(UInt8).zeros([3, 3, 2])
Tensor.random(0.0...1.0, [2, 2, 2])

ClTensor(Float32).zeros([3, 2, 2])
ClTensor(Float64).full([3, 4, 5], 3.8)
```

### Operations

A `Tensor` supports a wide variety of numerical operations.  Many of these
operations are provided by `Num.cr`, but any operation can be mapped across
one or more `Tensor`s using sophisticated broadcasted mapping routines.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor

# Convert a Tensor to a GPU backed Tensor
acl = a.astype(Float64).opencl

puts Num.add(a, b)

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

acl = a.opencl
bcl = a.opencl

puts acl.gemm(bcl).cpu

# [[7 , 10],
#  [15, 22]]

puts a.matmul(a)

# [[7 , 10],
#  [15, 22]]
```

### DataFrames

For more structured data, consider using a `DataFrame`

```crystal
df = DataFrame.from_items(
  foo: [1, 2, 3, 4, 5].to_tensor,
  bar: [2.73, 3.1, 4.8, 5.1, 3.2],
)

puts df

#    foo   bar
# 0    1  2.73
# 1    2   3.1
# 2    3   4.8
# 3    4   5.1
# 4    5   3.2
```

A `DataFrame` maintains types while still providing convenient
mapping and reduction operations

```crystal
puts df.c[:foo]

# 0  1
# 1  2
# 2  3
# 3  4
# 4  5
# Name: foo
# dtype: Int32

puts typeof(df.c[:foo])

# Series(Int32, Int32)

puts df.sum

# foo     15
# bar  18.93
```

With operations that broadcast across the `DataFrame`

```crystal
puts df.greater(df.mean)

#      foo    bar
# 0  false  false
# 1  false  false
# 2  false   true
# 3   true   true
# 4   true  false
```

Review the documentation for full implementation details, and if something is missing,
open an issue to add it!
