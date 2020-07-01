![num.cr](https://raw.githubusercontent.com/crystal-data/bottle/rename/static/numcr_logo.png)

[![Join the chat at https://gitter.im/crystal-data/bottle](https://badges.gitter.im/crystal-data/bottle.svg)](https://gitter.im/crystal-data/bottle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.com/crystal-data/num.cr.svg?branch=master)](https://travis-ci.com/crystal-data/num.cr)

Num.cr is the core shard needed for scientific computing with Crystal

- **Website:** https://crystal-data.github.io/num.cr
- **API Documentation:** https://crystal-data.github.io/num.cr/
- **Source code:** https://github.com/crystal-data/num.cr
- **Bug reports:** https://github.com/crystal-data/num.cr/issues

It provides:

- An n-dimensional `Tensor` data structure
- sophisticated reduction and accumulation routines
- data structures that can easily be passed to C libraries
- powerful linear algebra routines backed by LAPACK and BLAS

## Prerequisites

Num.cr relies on various BLAS and LAPACK implementations to provide performant linear
algebra routines.  The defaults are OpenBLAS and Lapacke.  On Darwin, the Accelerate
framework is used by default.  Num.cr also allows for computing run on a GPU, backed
by OpenCL, with BLAS operations backed by ClBlast.  Please review the relevant installation
instructions for these libraries if you would like to take advantage of their features.

## Usage

Num.cr provides data structures that facilitate element-wise operations,
accumulations, and reductions.  While some operations are backed by BLAS
and LaPACK, many vectorized operations use iteration written in Crystal.
The primary goal of this library was to provide a NumPy like interface in
Crystal, and performance will be revisited constantly as the library is
improved.

Contributing
------------
Num.cr requires help in many different ways to continue to grow as a shard.
Contributions such as high level documentation and code quality checks are needed just
as much as API enhancements.  If you are considering larger scale contributions
that extend beyond minor enhancements and bug fixes, contact Crystal Data
in order to be added to the organization to gain access to review and merge
permissions.

## Installation

Add this to your applications `shard.yml`

```
dependencies:
  num:
    github: crystal-data/num.cr
```

## Getting started with Num.cr

The core data structure of Num.cr is the `Tensor`, a flexible N-dimensional data structure.
There are many ways to initialize a `Tensor`, or a `ClTensor` if you need GPU accelerated
operations.

```crystal
[1, 2, 3].to_tensor
Tensor.from_array [1, 2, 3]
Tensor(UInt8).zeros([3, 3, 2])
Tensor.random(0.0...1.0, [2, 2, 2])

ClTensor(Float32).zeros_like(some_tensor)
ClTensor(Float64).full([3, 4, 5], 3.8)
```

Tensors support numerous mathematical operations and manipulation routines, including
operations requiring broadcasting to other shapes.

```crystal
a = [1, 2, 3, 4].to_tensor
b = [[3, 4, 5, 6], [5, 6, 7, 8]].to_tensor

# Convert a Tensor to a GPU backed Tensor
acl = a.astype(Float64).opencl

puts Num.add(a, b)

# a is broadcast to b's shape
# [[ 4,  6,  8, 10],
#  [ 6,  8, 10, 12]]

puts a.reshape(2, 2)

# [[1, 2],
#  [3, 4]]

puts b.transpose

# [[3, 5],
#  [4, 6],
#  [5, 7],
#  [6, 8]]

puts Num.cos(acl).cpu

# [0.540302 , -0.416147, -0.989992, -0.653644]
```

Both CPU backed Tensors and GPU backed Tensors support linear algebra routines
backed by optimized BLAS libraries.

```crystal
a = [[1, 2], [3, 4]].to_tensor.astype(Float32)
b = [[3, 4], [5, 6]].to_tensor.astype(Float32)

acl = a.opencl
bcl = b.opencl

puts a.inv

# [[-2  , 1   ],
#  [1.5 , -0.5]]

puts acl.matmul(bcl).cpu

# [[13, 16],
#  [29, 36]]
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

```
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
