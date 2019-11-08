![bottle](https://raw.githubusercontent.com/crystal-data/bottle/master/static/bottle_logo.png)

[![Build Status](https://travis-ci.org/crystal-data/bottle.svg?branch=master)](https://travis-ci.org/crystal-data/bottle) [![Join the chat at https://gitter.im/crystal-data/bottle](https://badges.gitter.im/crystal-data/bottle.svg)](https://gitter.im/crystal-data/bottle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Bottle is the core shard needed for scientific computing with Crystal

- **Website:** https://crystal-data.github.io/bottle
- **API Documentation:** https://crystal-data.github.io/bottle/
- **Source code:** https://github.com/crystaldata/bottle
- **Bug reports:** https://github.com/crystaldata/bottle/issues

It provides:

- An n-dimensional `Tensor` data structure
- sophisticated reduction and accumulation routines
- data structures that can easily be passed to C libraries
- powerful linear algebra routines backed by LAPACK and BLAS

## Prerequisites

Bottle relies on OpenBLAS and LAPACK for many underlying computations, and the
development packages must be present for Bottle to work correctly.

For Debian, use `libopenblas-dev` and `liblapack-dev`.  For other operating
systems review the relevant installation instructions for that OS.

## Usage

Bottle provides data structures that facilitate element-wise operations,
accumulations, and reductions.  While some operations are backed by BLAS
and LaPACK, many vectorized operations use iteration written in Crystal.
The primary goal of this library was to provide a NumPy like interface in
Crystal, and performance will be revisited constantly as the library is
improved.

Include `Bottle` to add `Tensor` to the top level namespace,
as well as provide access to `B`, Bottle's public API, which provides many
powerful numerical methods.

```crystal
include Bottle
```


```crystal
t = Tensor.new([2, 2, 3]) { |i| i }
```

```crystal
Tensor([[[ 0,  1],
         [ 2,  3]],

        [[ 4,  5],
         [ 6,  7]],

        [[ 8,  9],
         [10, 11]]])
```

```crystal
t + t
```

```crystal
Tensor([[[ 0,  2],
         [ 4,  6]],

        [[ 8, 10],
         [12, 14]],

        [[16, 18],
         [20, 22]]])
```

Bottle provides an n-dimensional Tensor for efficient data storage.
Slice and index these containers to return views into their data.

```crystal
t[1...] # =>
# [[[ 6,  7,  8],
#   [ 9, 10, 11]]]

t[0] # =>
# [[0, 1, 2],
#  [3, 4, 5]]

t[..., 1] # =>
# [[ 3,  4,  5],
#  [ 9, 10, 11]]

t[1..., 1...] # =>
# [[[ 9, 10, 11]]]
```

Make use of elementwise, outer, and accumulation operations.

```crystal
B.divide(t[0], t[1]) # =>
# [[    0.0,   0.143,    0.25],
#  [  0.333,     0.4,   0.455]]

B.multiply.outer(t[0, 0], t[0, 1]) # =>
# [[ 0,  0,  0],
#  [ 3,  4,  5],
#  [ 6,  8, 10]]
```

Use Linear Algebra Routines backed by BLAS and LAPACK

```crystal
B.dot(t[0, 0], t[0, 1]) # => 14.0

B.matmul(t[0, ..., ...2], t[1, ..., ...2]) # =>
# [[   9.0,   10.0],
#  [  54.0,   61.0]]

B.inv(t[0, ..., ...2]) # =>
# [[-1.333, 0.333],
#  [  1.0,   0.0]]

B.norm(t[0, 1]) # => 7.0710678118654755
```




Contributing
------------
Bottle requires help in many different ways to continue to grow as a shard.
Contributions such as high level documentation and code quality checks are needed just
as much as API enhancements.  If you are considering larger scale contributions
that extend beyond minor enhancements and bug fixes, contact Crystal Data
in order to be added to the organization to gain access to review and merge
permissions.
