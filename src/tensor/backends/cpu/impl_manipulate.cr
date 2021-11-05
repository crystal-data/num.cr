# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module Num
  extend self

  # Broadcasts a `Tensor` to a new shape.  Returns a read-only
  # view of the original `Tensor`.  Many elements in the `Tensor`
  # will refer to the same memory location, and the result is
  # rarely contiguous.
  #
  # Shapes must be broadcastable, and an error will be raised
  # if broadcasting fails.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   The shape of the desired output `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[AlwaysInline]
  def broadcast_to(arr : Tensor(U, CPU(U)), shape : Array(Int)) forall U
    strides = Num::Internal.strides_for_broadcast(arr.shape, arr.strides, shape)
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    flags &= ~Num::ArrayFlags::Write
    Tensor.new(arr.data, shape, strides, arr.offset, flags, U)
  end

  # `as_strided` creates a view into the `Tensor` given the exact strides
  # and shape. This means it manipulates the internal data structure
  # of a `Tensor` and, if done incorrectly, the array elements can point
  # to invalid memory and can corrupt results or crash your program.
  #
  # It is advisable to always use the original `strides` when
  # calculating new strides to avoid reliance on a contiguous
  # memory layout.
  #
  #
  # Furthermore, `Tensor`s created with this function often contain
  # self overlapping memory, so that two elements are identical.
  # Vectorized write operations on such `Tensor`s will typically be
  # unpredictable. They may even give different results for small,
  # large, or transposed `Tensor`s.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of the new `Tensor`
  # *strides*
  #   Strides of the new `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.as_strided([3, 3], [0, 1])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[AlwaysInline]
  def as_strided(arr : Tensor(U, CPU(U)), shape : Array(Int), strides : Array(Int)) : Tensor(U, CPU(U)) forall U
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    flags &= ~Num::ArrayFlags::Write
    Tensor.new(arr.data, shape, strides, arr.offset, flags, U)
  end

  # Expands a `Tensor`s dimensions n times by broadcasting
  # the shape and strides.  No data is copied, and the result
  # is a read-only view of the original `Tensor`
  #
  # Arguments
  # ---------
  # *n* : Int
  #   Number of dimensions to broadcast
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3].to_tensor
  # a.with_broadcast(2)
  #
  # # [[[1]],
  # #
  # #  [[2]],
  # #
  # #  [[3]]]
  # ```
  def with_broadcast(arr : Tensor(U, CPU(U)), n : Int) : Tensor(U, CPU(U)) forall U
    shape = arr.shape + [1] * n
    strides = arr.strides + [0] * n
    arr.as_strided(shape, strides)
  end

  # Brief description of expanddims
  #
  # Arguments
  # ---------
  # axis : Int
  #   Brief description of axis : Int
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def expand_dims(arr : Tensor(U, CPU(U)), axis : Int) : Tensor(U, CPU(U)) forall U
    shape = arr.shape.dup
    shape.insert(axis, 1)
    strides = arr.strides.dup
    strides.insert(axis, 0)
    arr.as_strided(shape, strides)
  end

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # Examples
  # ````````
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  #
  # x, y = a.broadcast(b)
  # x.shape # => [3, 3]
  # ```
  @[AlwaysInline]
  def broadcast(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
    if a.shape == b.shape
      return {a, b}
    end
    shape = Num::Internal.shape_for_broadcast(a, b)
    return {a.broadcast_to(shape), b.broadcast_to(shape)}
  end

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # Examples
  # ````````
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  #
  # x, y = a.broadcast(b)
  # x.shape # => [3, 3]
  # ````````
  @[AlwaysInline]
  def broadcast(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V)), c : Tensor(W, CPU(W))) forall U, V, W
    if a.shape == b.shape && a.shape == c.shape
      return {a, b, c}
    end
    shape = Num::Internal.shape_for_broadcast(a, b, c)
    return {a.broadcast_to(shape), b.broadcast_to(shape), c.broadcast_to(shape)}
  end

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # Arguments
  # ---------
  # *result_shape* : Array(Int)
  #   Result shape for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  @[AlwaysInline]
  def reshape(arr : Tensor(U, CPU(U)), shape : Array(Int)) forall U
    shape, strides = Num::Internal.strides_for_reshape(arr.shape, shape)
    flags = arr.flags.dup
    if arr.is_c_contiguous
      flags &= ~Num::ArrayFlags::OwnData
    else
      arr = arr.dup(Num::RowMajor)
    end
    Tensor(U, CPU(U)).new(arr.data, shape, strides, arr.offset, flags, U)
  end

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # Arguments
  # ---------
  # *result_shape* : Array(Int)
  #   Result shape for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  @[AlwaysInline]
  def reshape(arr : Tensor(U, CPU(U)), *shape : Int) forall U
    reshape(arr, shape.to_a)
  end

  # Flattens a `Tensor` to a single dimension.  If a view can be created,
  # the reshape operation will not copy data.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.flat # => [0, 1, 2, 3]
  # ```
  @[AlwaysInline]
  def flat(arr : Tensor(U, CPU(U))) forall U
    reshape(arr, -1)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # Arguments
  # ---------
  # *arr* : Tensor
  #   Tensor to permute
  # *source* : Array(Int)
  #   Original positions of axes to move
  # *destination*
  #   Destination positions to permute axes
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # moveaxis(a, [0], [-1]).shape # => 4, 5, 3
  # ```
  @[AlwaysInline]
  def move_axis(arr : Tensor(U, CPU(U)), source : Array(Int), destination : Array(Int)) forall U
    axes = Num::Internal.move_axes_for_transpose(arr.rank, source, destination)
    transpose(arr, axes)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # Arguments
  # ---------
  # *arr* : Tensor
  #   Tensor to permute
  # *source* : Array(Int)
  #   Original positions of axes to move
  # *destination*
  #   Destination positions to permute axes
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # moveaxis(a, [0], [-1]).shape # => 4, 5, 3
  # ```
  @[AlwaysInline]
  def move_axis(arr : Tensor(U, CPU(U)), source : Int, destination : Int) forall U
    moveaxis(arr, [source], [destination])
  end

  # Permutes two axes of a `Tensor`.  This will always create a view
  # of the permuted `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Int
  #   First axis of permutation
  # *b* : Int
  #   Second axis of permutation
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.swap_axes(2, 0)
  #
  # # [[[ 0,  6, 12, 18]
  # #   [ 2,  8, 14, 20]
  # #   [ 4, 10, 16, 22]]
  # #
  # #  [[ 1,  7, 13, 19]
  # #   [ 3,  9, 15, 21]
  # #   [ 5, 11, 17, 23]]]
  # ```
  @[AlwaysInline]
  def swap_axes(arr : Tensor(U, CPU(U)), a : Int, b : Int) forall U
    axes = Num::Internal.swap_axes_for_transpose(arr.rank, a, b)
    transpose(arr, axes)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # Arguments
  # ---------
  # *axes* : Array(Int)
  #   New ordering of axes for the permuted `Tensor`.  If empty,
  #   a full transpose will occur
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.transpose([2, 0, 1])
  #
  # # [[[ 0,  2,  4],
  # #   [ 6,  8, 10],
  # #   [12, 14, 16],
  # #   [18, 20, 22]],
  # #
  # #  [[ 1,  3,  5],
  # #   [ 7,  9, 11],
  # #   [13, 15, 17],
  # #   [19, 21, 23]]]
  # ```
  @[AlwaysInline]
  def transpose(arr : Tensor(U, CPU(U)), axes : Array(Int) = [] of Int32) forall U
    shape, strides = Num::Internal.shape_and_strides_for_transpose(arr.shape, arr.strides, axes)
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    Tensor(U, CPU(U)).new(arr.data, shape, strides, arr.offset, U)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # Arguments
  # ---------
  # *axes* : Array(Int)
  #   New ordering of axes for the permuted `Tensor`.  If empty,
  #   a full transpose will occur
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.transpose([2, 0, 1])
  #
  # # [[[ 0,  2,  4],
  # #   [ 6,  8, 10],
  # #   [12, 14, 16],
  # #   [18, 20, 22]],
  # #
  # #  [[ 1,  3,  5],
  # #   [ 7,  9, 11],
  # #   [13, 15, 17],
  # #   [19, 21, 23]]]
  # ```
  @[AlwaysInline]
  def transpose(arr : Tensor(U, CPU(U)), *args : Int) forall U
    transpose(arr, args.to_a)
  end

  # Repeat elements of a `Tensor`, treating the `Tensor`
  # as flat
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Object to repeat
  # - `n` : Int
  #   Number of times to repeat
  #
  # Examples
  # ```
  # a = [1, 2, 3]
  # Num.repeat(a, 2) # => [1, 1, 2, 2, 3, 3]
  # ```
  @[AlwaysInline]
  def repeat(a : Tensor(U, CPU(U)), n : Int) forall U
    result = Tensor(U, CPU(U)).new([a.size * n])
    iter = result.each
    Num::Internal.repeat_inner(a, n) do |value|
      iter.next.value = value
    end
    result
  end

  # Repeat elements of a `Tensor` along an axis
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Object to repeat
  # - `n` : Int
  #   Number of times to repeat
  # - `axis` : Int
  #   Axis along which to repeat
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # Num.repeat(a, 2, 1)
  #
  # # [[1, 1, 2, 2, 3, 3],
  # #  [4, 4, 5, 5, 6, 6]]
  # ```
  @[AlwaysInline]
  def repeat(a : Tensor(U, CPU(U)), n : Int, axis : Int) forall U
    shape = a.shape.dup
    shape[axis] *= n
    result = Tensor(U, CPU(U)).new(shape)
    iter = each_axis(result, axis.to_i)
    each_axis(a, axis) do |ax|
      n.times do
        iter.next[...] = ax
      end
    end
    result
  end

  # Tile elements of a `Tensor`
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to tile
  # - `n` : Int
  #   Number of times to tile
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  @[AlwaysInline]
  def tile(a : Tensor(U, CPU(U)), n : Int) forall U
    d = a.rank > 1 ? [1] * (a.rank - 1) + [n] : [1]
    Num::Internal.tile_inner(a, d)
  end

  # Tile elements of a `Tensor`
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to tile
  # - `n` : Int
  #   Number of times to tile
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  @[AlwaysInline]
  def tile(a : Tensor(U, CPU(U)), n : Array(Int)) forall U
    n = n.size < a.rank ? [1] * (a.rank - n.size) + n : n
    Num::Internal.tile_inner(a, n)
  end

  # Flips a `Tensor` along all axes, returning a view
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to flip
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a)
  #
  # # [[6, 5, 4],
  # #  [3, 2, 1]]
  # ```
  @[AlwaysInline]
  def flip(a : Tensor(U, CPU(U))) forall U
    i = [{..., -1}] * a.rank
    a[i]
  end

  # Flips a `Tensor` along an axis, returning a view
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to flip
  # - `axis` : Int
  #   Axis to flip
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a, 1)
  #
  # # [[3, 2, 1],
  # #  [6, 5, 4]]
  # ```
  @[AlwaysInline]
  def flip(a : Tensor(U, CPU(U)), axis : Int) forall U
    s = (0...a.rank).map do |i|
      i == axis ? {..., -1} : (...)
    end
    a[s]
  end
end
