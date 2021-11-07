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

class Tensor(T, S)
  # Broadcasts a `Tensor` to a new shape.  Returns a read-only
  # view of the original `Tensor`.  Many elements in the `Tensor`
  # will refer to the same memory location, and the result is
  # rarely contiguous.
  #
  # Shapes must be broadcastable, and an error will be raised
  # if broadcasting fails.
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - The shape of the desired output `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def broadcast_to(shape : Array(Int)) : Tensor(T, S)
    Num.broadcast_to(self, shape)
  end

  # Expands a `Tensor`s dimensions n times by broadcasting
  # the shape and strides.  No data is copied, and the result
  # is a read-only view of the original `Tensor`
  #
  # ## Arguments
  #
  # * n : `Int` - Number of dimensions to broadcast
  #
  # ## Examples
  #
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
  def with_broadcast(n : Int) : Tensor(T, S)
    Num.with_broadcast(self, n)
  end

  # Expands the dimensions of a `Tensor`, along a single axis
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis on which to expand dimensions
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.expand_dims(1)
  #
  # # [[[0, 1]],
  # #
  # # [[2, 3]]]
  # ```
  def expand_dims(axis : Int) : Tensor(T, S)
    Num.expand_dims(self, axis)
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of the new `Tensor`
  # * strides : `Array(Int)` - Strides of the new `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.as_strided([3, 3], [0, 1])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def as_strided(shape : Array(Int), strides : Array(Int)) : Tensor(T, S)
    Num.as_strided(self, shape, strides)
  end

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # ## Arguments
  #
  # * other : `Tensor` - RHS of the broadcast
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  #
  # x, y = a.broadcast(b)
  # x.shape # => [3, 3]
  # ```
  def broadcast(other : Tensor(U, V)) forall U, V
    Num.broadcast(self, other)
  end

  # Broadcasts three `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the three Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # ## Arguments
  #
  # * other1 : `Tensor` - `Tensor` to broadcast
  # * other2 : `Tensor` - Additional `Tensor` to broadcast
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  # c = Tensor.new([3, 3, 3]) { |i| i }
  #
  # x, y, z = a.broadcast(b, c)
  # x.shape # => [3, 3, 3]
  # ```
  def broadcast(other1 : Tensor(U, V), other2 : Tensor(W, X)) forall U, V, W, X
    Num.broadcast(self, other1, other2)
  end

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # ## Arguments
  #
  # * result_shape : `Array(Int)` - Result shape for the `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  def reshape(shape : Array(Int)) : Tensor(T, S)
    Num.reshape(self, shape)
  end

  # :ditto:
  def reshape(*shape : Int) : Tensor(T, S)
    Num.reshape(self, *shape)
  end

  # Flattens a `Tensor` to a single dimension.  If a view can be created,
  # the reshape operation will not copy data.
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.flat # => [0, 1, 2, 3]
  # ```
  def flat : Tensor(T, S)
    Num.flat(self)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # ## Arguments
  #
  # * source : `Array(Int)` - Original positions of axes to move
  # * destination : `Array(Int)` - Destination positions to permute axes
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # a.moveaxis([0], [-1]).shape # => 4, 5, 3
  # ```
  def move_axis(source : Array(Int), destination : Array(Int)) : Tensor(T, S)
    Num.move_axis(self, source, destination)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # ## Arguments
  #
  # * source : `Int` - Original position of axis to move
  # * destination : `Int` - Destination position of axis
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # a.moveaxis(0, -1).shape # => 4, 5, 3
  # ```
  def move_axis(source : Int, destination : Int) : Tensor(T, S)
    Num.move_axis(self, source, destination)
  end

  # Permutes two axes of a `Tensor`.  This will always create a view
  # of the permuted `Tensor`
  #
  # ## Arguments
  #
  # * a : `Int` - First axis of permutation
  # * b : `Int` - Second axis of permutation
  #
  # ## Examples
  #
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
  def swap_axes(a : Int, b : Int) : Tensor(T, S)
    Num.swap_axes(self, a, b)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # ## Arguments
  #
  # * axes : `Array(Int)` - New ordering of axes for the permuted `Tensor`.
  #   If empty, a full transpose will occur
  #
  # ## Examples
  #
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
  def transpose(axes : Array(Int) = [] of Int32) : Tensor(T, S)
    Num.transpose(self, axes)
  end

  # :ditto:
  def transpose(*axes : Int) : Tensor(T, S)
    Num.transpose(self, *axes)
  end

  # Repeat elements of a `Tensor`, treating the `Tensor`
  # as flat
  #
  # ## Arguments
  #
  # * n : `Int` - Number of times to repeat
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # a.repeat(2) # => [1, 1, 2, 2, 3, 3]
  # ```
  def repeat(n : Int) : Tensor(T, S)
    Num.repeat(self, n)
  end

  # Repeat elements of a `Tensor` along an axis
  #
  # ## Arguments
  #
  # * n : Int - Number of times to repeat
  # * axis : `Int` - Axis along which to repeat
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # a.repeat(2, 1)
  #
  # # [[1, 1, 2, 2, 3, 3],
  # #  [4, 4, 5, 5, 6, 6]]
  # ```
  def repeat(n : Int, axis : Int) : Tensor(T, S)
    Num.repeat(self, n, axis)
  end

  # Tile elements of a `Tensor`
  #
  # ## Arguments
  #
  # * n : `Int` - Number of times to tile
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts a.tile(2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  def tile(n : Int) : Tensor(T, S)
    Num.tile(self, n)
  end

  # Tile elements of a `Tensor`
  #
  # ## Arguments
  #
  # * n : `Int` - Number of times to tile
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  def tile(n : Array(Int)) : Tensor(T, S)
    Num.tile(self, n)
  end

  # Flips a `Tensor` along all axes, returning a view
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # a.flip
  #
  # # [[6, 5, 4],
  # #  [3, 2, 1]]
  # ```
  def flip : Tensor(T, S)
    Num.flip(self)
  end

  # Flips a `Tensor` along an axis, returning a view
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis to flip
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # a.flip(1)
  #
  # # [[3, 2, 1],
  # #  [6, 5, 4]]
  # ```
  def flip(axis : Int) : Tensor(T, S)
    Num.flip(self, axis)
  end

  # Returns a view of the diagonal of a `Tensor`.  This method only works
  # for two-dimensional arrays.
  #
  # TODO: Implement views for offset diagonals
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new(3, 3) { |i, _| i }
  # a.diagonal # => [0, 1, 2]
  # ```
  def diagonal : Tensor(T, S)
    Num.diagonal(self)
  end
end
