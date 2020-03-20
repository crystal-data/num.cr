***************
What is Num.cr?
***************

Num.cr is the core numerical computing shard for Crystal Lang.
It implements an N-Dimensional tensor for homogenous data, and provides
a large collection of routines for fast operations on these tensors.
These operations include mathematical, logical, sorting, selecting, basic linear
algebra, basic statistical operations, and more.

At the core of the Num.cr shard is the `Tensor` class.  This allows
for N-Dimensional representation of data.  There are key differences between
the `Tensor`, and a basic `Crystal` array.

- Tensors have a fixed size at creation.  Operations that reduce or re-size
  a Tensor copy data to create a new Tensor.

- Elements in a Tensor must be homogenous, so that they take up the same amount
  of space in memory.  Tensors are navigated using strides, and strides must
  always be consistent to support fast operations.

- By taking advantage of LLVM optimizations and low level C libraries, Tensors
  support advanced mathematical operations on large numbers of data.  Since
  Crystal is quite fast, and arrays do not have to worry about strides and
  contiguous memory, some operations on 1-dimensional data may be faster
  using standard library arrays.  However, the power of Num.cr comes when
  data must be represented and manipulated in many dimensions.
