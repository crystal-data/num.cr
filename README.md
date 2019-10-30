![bottle](https://raw.githubusercontent.com/crystal-data/bottle/master/static/bottle_logo.png)

[![Build Status](https://travis-ci.org/crystal-data/bottle.svg?branch=master)](https://travis-ci.org/crystal-data/bottle) [![Join the chat at https://gitter.im/crystal-data/bottle](https://badges.gitter.im/crystal-data/bottle.svg)](https://gitter.im/crystal-data/bottle?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Bottle is the core shard needed for scientific computing with Crystal

- **Website:** https://www.crystaldata.org
- **API Documentation:** http://crystaldata.org/bottle
- **Source code:** https://github.com/crystaldata/bottle
- **Bug reports:** https://github.com/crystaldata/bottle/issues

It provides:

- efficient 1 and 2-dimensional data structures
- sophisticated reduction and accumulation routines
- data structures that can easily be passed to C libraries
- powerful linear algebra routines backed by LAPACK and BLAS

## Usage

Bottle provides data structures that facilitate element-wise operations,
accumulations, and reductions.  While some operations are backed by BLAS
and LaPACK, many vectorized operations use iteration written in Crystal.
The primary goal of this library was to provide a NumPy like interface in
Crystal, and performance will be revisited constantly as the library is
improved.

Include `Bottle` to add `Matrix` and `Tensor` to the top level namespace,
as well as provide access to `B`, Bottle's public API, which provides many
powerful numerical methods.

```crystal
include Bottle

t = Tensor.new [1, 2, 3]

B.add(t, t) # => Tensor[  2  4  6]
```

Bottle provides `Tensor` and `Matrix` classes for efficient data storage.

```crystal

t = Tensor.new [1.0, 2.0, 3.0, 4.0, 5.0]
m = Matrix.new [[1.0, 2.0], [3.0, 4.0]]
```

Slice and index these containers to return views into their data.

```crystal
t[1...] # => Tensor[     2.0     3.0     4.0     5.0]

m[0] # => Tensor[     1.0     2.0]

m[..., 1] # => Tensor[     2.0     4.0]

m[1..., 1...] # => Matrix[[     4.0]]
```

Make use of elementwise, outer, and accumulation operations.

```crystal
B.divide(t[...2], m[1]) # => Tensor[   0.333     0.5]

B.multiply.outer(t, m[1]) # =>
# Matrix[[      3.0      4.0]
#        [      6.0      8.0]
#        [      9.0     12.0]
#        [     12.0     16.0]
#        [     15.0     20.0]]

B.add.accumulate(t) # => Tensor[      1.0      3.0      6.0     10.0     15.0]
```

Use Linear Algebra Routines backed by BLAS and LAPACK

```crystal
B.dot(t, t) # => 55.0

B.matmul(m, m) # =>
# Matrix[[      7.0     10.0]
#        [     15.0     22.0]]

B.inv(m) # =>
# Matrix[[    -2.0     1.0]
#        [     1.5    -0.5]]

B.norm(t) # => 7.416198487095663
```

Basic broadcasting operations are supported, and will be enhanced when the library
supports N-Dimensional Tensors.

```crystal
slice = t[2...4] # => Tensor[   3.0   4.0]

# row-wise broadcast
slice[nil] * m # =>
# Matrix[[      3.0      8.0]
#        [      9.0     16.0]]

slice[..., nil] * m # =>
# Matrix[[      3.0      6.0]
#        [     12.0     16.0]]
```




Contributing
------------
Bottle requires help in many different ways to continue to grow as a shard.
Contributions such as high level documentation and code quality checks are needed just
as much as API enhancements.  If you are considering larger scale contributions
that extend beyond minor enhancements and bug fixes, contact Crystal Data
in order to be added to the organization to gain access to review and merge
permissions.
