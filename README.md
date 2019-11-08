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
t[1...]
```

```crystal
Tensor([[[ 4,  5],
         [ 6,  7]],

        [[ 8,  9],
         [10, 11]]])
```

```crystal
t[0]
```

```crystal
Tensor([[0, 1],
        [2, 3]])
```

```crystal
t[..., 1] # =>
```

```crystal
Tensor([[ 2,  3],
        [ 6,  7],
        [10, 11]])
```

```crystal
t[..., 1...2, 1...2]
```

```crystal
Tensor([[[ 3]],

        [[ 7]],

        [[11]]])
```

Make use of elementwise, outer, and accumulation operations.

```crystal
t[0] / t[-1]
```

```crystal
Tensor([[    0.0,   0.111],
        [    0.2,   0.273]])
```

```crystal
B.multiply.outer(t[...2, 1], t[1..., -1])
```

```crystal
Tensor([[[[12, 14],
          [18, 21]],

         [[20, 22],
          [30, 33]]],


        [[[36, 42],
          [42, 49]],

         [[60, 66],
          [70, 77]]]])
```

Use Linear Algebra Routines backed by BLAS and LAPACK

```crystal
B.matmul(t[0], t[1]) # =>
```

```crystal
Tensor([[   6.0,    7.0],
        [  26.0,   31.0]])
```

```crystal
B.inv(t[0, ..., ...2])
```

```crystal
Tensor([[ -1.5,   0.5],
        [  1.0,   0.0]])
```




Contributing
------------
Bottle requires help in many different ways to continue to grow as a shard.
Contributions such as high level documentation and code quality checks are needed just
as much as API enhancements.  If you are considering larger scale contributions
that extend beyond minor enhancements and bug fixes, contact Crystal Data
in order to be added to the organization to gain access to review and merge
permissions.
