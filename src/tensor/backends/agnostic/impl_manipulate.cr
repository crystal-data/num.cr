# Copyright (c) 2021 Crystal Data Contributors
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
  # * arr : `Tensor` - `Tensor` to broadcast
  # * shape : `Array(Int)` - The shape of the desired output `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[Inline]
  def broadcast_to(arr : Tensor(U, V), shape : Array(Int)) forall U, V
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
  # ## Arguments
  #
  # * arr : `Tensor` - `Tensor` to broadcast
  # * shape : `Array(Int)` - Shape of broadcasted `Tensor`
  # * strides : `Array(Int)` - Strides of broadcasted `Tensor`
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
  @[Inline]
  def as_strided(arr : Tensor(U, V), shape : Array(Int), strides : Array(Int)) : Tensor(U, V) forall U, V
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    flags &= ~Num::ArrayFlags::Write
    Tensor.new(arr.data, shape, strides, arr.offset, flags, U)
  end

  # Expands a `Tensor`s dimensions n times by broadcasting
  # the shape and strides.  No data is copied, and the result
  # is a read-only view of the original `Tensor`
  #
  # ## Arguments
  #
  # * arr : `Tensor` - `Tensor` to broadcast
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
  @[Inline]
  def with_broadcast(arr : Tensor(U, V), n : Int) : Tensor(U, V) forall U, V
    shape = arr.shape + [1] * n
    strides = arr.strides + [0] * n
    arr.as_strided(shape, strides)
  end

  # Expands a `Tensor` along an `axis`
  #
  # ## Arguments
  #
  # * arr : `Tensor` - `Tensor` to expand
  # * axis : `Int` - `Axis` along which to expand
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # a.expand_dims(0) # => [[1, 2, 3]]
  # ```
  @[Inline]
  def expand_dims(arr : Tensor(U, V), axis : Int) : Tensor(U, V) forall U, V
    shape = arr.shape.dup
    shape.insert(axis, 1)
    strides = arr.strides.dup
    strides.insert(axis, 0)
    arr.as_strided(shape, strides)
  end

  # Join a sequence of `Tensor`s along an existing axis.  The `Tensor`s
  # must have the same shape for all axes other than the axis of
  # concatenation
  #
  # ## Arguments
  #
  # * arrs : `Array(Tensor)` - `Tensor`s to concatenate
  # * axis : `Int` - Axis of concatenation, negative axes are allowed
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # b = Tensor.from_array [4, 5, 6]
  # Num.concat([a, b], 0) # => [1, 2, 3, 4, 5, 6]
  #
  # c = Tensor.new([3, 2, 2]) { |i| i / 2 }
  # Num.concat([c, c, c], -1)
  #
  # # [[[0  , 0.5, 0  , 0.5, 0  , 0.5],
  # #  [1  , 1.5, 1  , 1.5, 1  , 1.5]],
  # #
  # # [[2  , 2.5, 2  , 2.5, 2  , 2.5],
  # #  [3  , 3.5, 3  , 3.5, 3  , 3.5]],
  # #
  # # [[4  , 4.5, 4  , 4.5, 4  , 4.5],
  # #  [5  , 5.5, 5  , 5.5, 5  , 5.5]]]
  # ```
  @[Inline]
  def concatenate(arrs : Array(Tensor(U, V)), axis : Int) forall U, V
    Num::Internal.assert_min_dimension(arrs, 1)
    shape = arrs[0].shape.dup

    axis = Num::Internal.clip_axis(axis, shape.size)
    shape[axis] = 0

    shape = Num::Internal.concat_shape(arrs, axis, shape)
    result = arrs[0].class.new(shape)

    lo = [0] * result.rank
    hi = shape.dup
    hi[axis] = 0

    arrs.each do |a|
      if a.shape[axis] != 0
        hi[axis] += a.shape[axis]
        ranges = lo.zip(hi).map do |i, j|
          i...j
        end
        result[ranges] = a
        lo[axis] = hi[axis]
      end
    end
    result
  end

  # Join a sequence of `Tensor`s along an existing axis.  The `Tensor`s
  # must have the same shape for all axes other than the axis of
  # concatenation
  #
  # ## Arguments
  #
  # * arrs : `Tuple(Tensor)` - `Tensor`s to concatenate
  # * axis : `Int` - Axis of concatenation, negative axes are allowed
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # b = Tensor.from_array [4, 5, 6]
  # Num.concat([a, b], 0) # => [1, 2, 3, 4, 5, 6]
  #
  # c = Tensor.new([3, 2, 2]) { |i| i / 2 }
  # Num.concat([c, c, c], -1)
  #
  # # [[[0  , 0.5, 0  , 0.5, 0  , 0.5],
  # #  [1  , 1.5, 1  , 1.5, 1  , 1.5]],
  # #
  # # [[2  , 2.5, 2  , 2.5, 2  , 2.5],
  # #  [3  , 3.5, 3  , 3.5, 3  , 3.5]],
  # #
  # # [[4  , 4.5, 4  , 4.5, 4  , 4.5],
  # #  [5  , 5.5, 5  , 5.5, 5  , 5.5]]]
  # ```
  @[Inline]
  def concatenate(*arrs : Tensor(U, V), axis : Int) forall U, V
    concatenate(arrs.to_a, axis)
  end

  # Stack an array of `Tensor`s in sequence row-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # ## Arguments
  #
  # * arrs : `Array(Tensor)` - `Tensor`s to concatenate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.vstack([a, a])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[Inline]
  def vstack(arrs : Array(Tensor(U, V))) forall U, V
    concatenate(arrs, 0)
  end

  # Stack an array of `Tensor`s in sequence row-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # ## Arguments
  #
  # * arrs : `Tuple(Tensor)` - `Tensor`s to concatenate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.vstack([a, a])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[Inline]
  def vstack(*arrs : Tensor(U, V)) forall U, V
    concatenate(arrs.to_a, 0)
  end

  # Stack an array of `Tensor`s in sequence column-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # For one dimensional `Tensor`s, this will still stack along the
  # first axis
  #
  # ## Arguments
  #
  # * arrs : `Array(Tensor)` - `Tensor`s to concatenate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.h_concat([a, a]) # => [1, 2, 3, 1, 2, 3]
  #
  # b = [[1, 2], [3, 4]].to_tensor
  # Num.h_concat([b, b])
  #
  # # [[1, 2, 1, 2],
  # #  [3, 4, 3, 4]]
  # ```
  @[Inline]
  def hstack(arrs : Array(Tensor(U, V))) forall U, V
    concatenate(arrs, 1)
  end

  # Stack an array of `Tensor`s in sequence column-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # For one dimensional `Tensor`s, this will still stack along the
  # first axis
  #
  # ## Arguments
  #
  # * arrs : `Tuple(Tensor)` - `Tensor`s to concatenate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.h_concat([a, a]) # => [1, 2, 3, 1, 2, 3]
  #
  # b = [[1, 2], [3, 4]].to_tensor
  # Num.h_concat([b, b])
  #
  # # [[1, 2, 1, 2],
  # #  [3, 4, 3, 4]]
  # ```
  @[Inline]
  def hstack(*arrs : Tensor(U, V)) forall U, V
    concatenate(arrs.to_a, 1)
  end
end
