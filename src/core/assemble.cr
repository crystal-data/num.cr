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
require "./macros"
require "./common"

module Num
  extend self

  # Join a sequence of arrays along an existing axis.
  # The arrays must have the same shape, except in the dimension corresponding
  # to axis (the first, by default).
  #
  # Parameters:
  # - alist : Array(BaseArray(U))
  #   The arrays to concatenate
  # - axis : Int32
  #   The axis along which to concatenate
  #
  # Return:
  # - BaseArray(U) - The concatenated arrays
  #
  # Example:
  # ```
  # a = Num.zeros([3, 3])
  # b = Num.ones([3, 3])
  # puts Num.concatenate([a, b], 1)
  # ```
  #
  # Output
  # ```
  # Tensor([[0, 0, 0, 1, 1, 1],
  #         [0, 0, 0, 1, 1, 1],
  #         [0, 0, 0, 1, 1, 1]])
  # ```
  def concatenate(alist : Array(BaseArray(U)), axis : Int32) forall U
    # This particular method does not allow zero dimensional items, even
    # if they can be upcast, since they can't match a shape off axis, and
    # concatenation must occur along an existing axis.
    NumInternal.raise_zerod alist
    newshape = alist[0].shape.dup

    # Just a check for negative axes, so that negative axes can be
    # inferred.
    axis = NumInternal.clip_axis axis, newshape.size
    newshape[axis] = 0

    # All arrays must share a shape of the axis of concatenation
    shape = NumInternal.assert_shape_off_axis(alist, axis, newshape)
    ret = alist[0].class.new(newshape)
    lo = [0] * newshape.size
    hi = shape.dup
    hi[axis] = 0

    # Basic slice looping and assignment, then updating the offset.
    alist.each do |a|
      if a.shape[axis] != 0
        hi[axis] += a.shape[axis]
        ranges = lo.zip(hi).map { |i, j| i...j }
        ret[ranges] = a
        lo[axis] = hi[axis]
      end
    end
    ret
  end

  # Concatenates an array of one dimensional tensors.
  def concatenate(alist : Array(BaseArray(U))) forall U
    totalsize = alist.reduce(0) { |i, j| i + j.size }
    ret = alist[0].class.new([totalsize])
    offset = 0
    alist.each do |a|
      ret[offset...(offset + a.size)] = a
      offset += a.size
    end
    ret
  end

  # Concatenates a list of `Tensor`s along axis 0
  def vstack(alist : Array(BaseArray(U))) forall U
    alist = alist.map { |t| atleast_2d(t) }
    concatenate(alist, 0)
  end

  # Concatenates a list of `Tensor`s along axis 1
  def hstack(alist : Array(BaseArray(U))) forall U
    alist = alist.map { |e| Num.atleast_1d(e) }
    NumInternal.less_than_3d(alist)
    if alist.all? { |t| t.ndims == 1 }
      concatenate(alist)
    else
      concatenate(alist, 1)
    end
  end

  def dstack(alist : Array(BaseArray(U))) forall U
    alist = alist.map { |e| Num.atleast_1d(e) }
    first = alist[0]
    shape = first.shape
    NumInternal.assert_shape(shape, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([1, a.size, 1])
      end
      concatenate(alist, 2)
    when 2
      alist = alist.map do |a|
        a.reshape(a.shape + [1])
      end
      concatenate(alist, 2)
    else
      raise NumInternal::ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end

  def column_stack(alist : Array(BaseArray(U))) forall U
    alist = alist.map { |e| Num.atleast_1d(e) }
    first = alist[0]
    shape = first.shape
    NumInternal.assert_shape(shape, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([a.size, 1])
      end
      concatenate(alist, 1)
    when 2
      concatenate(alist, 1)
    else
      raise NumInternal::ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end

  def stack(alist : Array(BaseArray(U)), axis : Int32 = 0) forall U
    first = alist[0]
    shape = first.shape
    assert_shape(shape, alist)
    if alist.all? { |e| e.ndims == 0 }
      return concatenate(alist)
    end
    assert_all_1d alist
    expanded_arrays = alist.map { |e| e.bc(axis) }
    concatenate(expanded_arrays, axis)
  end

  def atleast_1d(inp : Number)
    Tensor.new([1]) { |_| inp }
  end

  def atleast_1d(inp : Array)
    Tensor.from_array inp
  end

  def atleast_1d(inp : Tensor)
    if inp.ndims == 0
      Tensor.new([1]) { |_| inp.value }
    else
      inp
    end
  end

  def atleast_2d(inp : Number)
    Tensor.new([1, 1]) { |_| inp }
  end

  def atleast_2d(inp : Array)
    t = Tensor.from_array inp
    atleast_2d(t)
  end

  def atleast_2d(inp : Tensor)
    if inp.ndims > 1
      inp
    elsif inp.ndims == 0
      Tensor.new([1, 1]) { |_| inp.value }
    else
      inp.reshape([1, inp.size])
    end
  end

  def atleast_3d(inp : Number)
    Tensor.new([1, 1, 1]) { |_| inp }
  end

  def atleast_3d(inp : Tensor)
    if inp.ndims > 2
      inp
    else
      dim = 3 - inp.ndims
      newshape = [1] * dim + inp.shape
      inp.reshape(newshape)
    end
  end
end
