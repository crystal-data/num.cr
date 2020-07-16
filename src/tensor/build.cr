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

require "./tensor"
require "./extensions/*"

module Num
  extend self

  # Join a sequence of `Tensor`s along an existing axis.  The `Tensor`s
  # must have the same shape for all axes other than the axis of
  # concatenation
  #
  # Arguments
  # ---------
  # *t_array* : Array(Tensor | Enumerable)
  #   Array of items to concatenate.  All elements
  #   will be cast to `Tensor`, so arrays can be passed here, but
  #   all inputs must have the same generic type.  Union types
  #   are not allowed
  # *axis* : Int
  #   Axis of concatenation, negative axes are allowed
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
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
  def concat(t_array : Array(Tensor | Enumerable), axis : Int)
    tensor_array = t_array.map &.to_tensor
    assert_min_dimension(tensor_array, 1)
    new_shape = tensor_array[0].shape.dup

    axis = clip_axis(axis, new_shape.size)
    new_shape[axis] = 0

    shape = concat_shape(tensor_array, axis, new_shape)
    t = tensor_array[0].class.new(shape)

    lo = [0] * t.rank
    hi = shape.dup
    hi[axis] = 0

    tensor_array.each do |a|
      if a.shape[axis] != 0
        hi[axis] += a.shape[axis]
        ranges = lo.zip(hi).map do |i, j|
          i...j
        end
        t[ranges] = a
        lo[axis] = hi[axis]
      end
    end
    t
  end

  # :ditto:
  def concat(*t_args : Tensor | Enumerable, axis : Int)
    concat(t_args.to_a, axis)
  end

  # Join a sequence of one dimensional `Tensor`s along the first axis,
  # creating a one-dimensional output
  #
  # Arguments
  # ---------
  # *t_array* : Array(Tensor | Enumerable)
  #   Array of items to concatenate.  All elements
  #   will be cast to `Tensor`, so arrays can be passed here, but
  #   all inputs must have the same generic type.  Union types
  #   are not allowed
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # b = Tensor.from_array [4, 5, 6]
  # Num.concat([a, b]) # => [1, 2, 3, 4, 5, 6]
  # ```
  def concat(t_array : Array(Tensor | Enumerable))
    tensor_array = t_array.map &.to_tensor
    total_size = tensor_array.reduce(0) do |i, j|
      i + j.size
    end

    offset = 0
    t = tensor_array[0].class.new([total_size])
    tensor_array.each do |a|
      t[offset...(offset + a.size)] = a
      offset += a.size
    end
    t
  end

  # :ditto:
  def concat(*t_args : Tensor | Enumerable)
    concat(t_args.to_a)
  end

  # Stack an array of `Tensor`s in sequence row-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # Arguments
  # *t_array* : Array(Tensor | Enumerable)
  #   `Tensor`s to concatenate
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.v_concat([a, a])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def v_concat(t_array : Array(Tensor | Enumerable))
    tensor_array = t_array.map &.to_tensor.with_dims(2)
    concat(tensor_array, 0)
  end

  # :ditto:
  def v_concat(*t_args : Tensor | Enumerable)
    v_concat(t_args.to_a)
  end

  # Stack an array of `Tensor`s in sequence column-wise.  While this
  # method can take `Tensor`s with any number of dimensions, it makes
  # the most sense with rank <= 3
  #
  # For one dimensional `Tensor`s, this will still stack along the
  # first axis
  #
  # Arguments
  # *t_array* : Array(Tensor | Enumerable)
  #   `Tensor`s to concatenate
  #
  # Examples
  # --------
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
  def h_concat(t_array : Array(Tensor | Enumerable))
    tensor_array = t_array.map &.to_tensor.with_dims(1)
    if tensor_array[0].rank == 1
      concat(tensor_array, 0)
    else
      concat(tensor_array, 1)
    end
  end

  # :ditto:
  def h_concat(*t_args : Tensor | Enumerable)
    h_concat(t_args.to_a)
  end

  # :nodoc:
  private def assert_min_dimension(ts : Array(Tensor), min : Int)
    unbounded = ts.any? do |t|
      t.rank < min
    end
    if unbounded
      raise Num::Internal::ShapeError.new("Wrong number of dimensions")
    end
  end

  # :nodoc:
  private def all_shapes_equal(shapes : Array(Array(Int)))
    s0 = shapes[0]
    shapes[1...].each do |s|
      unless s0 == s
        raise Num::Internal::ShapeError.new("All inputs must share a shape")
      end
    end
  end

  # :nodoc:
  def clip_axis(axis, size)
    if axis < 0
      axis += size
    end
    if axis < 0 || axis > size
      raise Num::Internal::AxisError.new("Axis out of range")
    end
    axis
  end

  # :nodoc:
  private def concat_shape(ts : Array(Tensor), axis : Int, shape : Array(Int))
    rank = shape.size
    ts.each do |t|
      if t.rank != rank
        raise Num::Internal::ShapeError.new(
          "All inputs must share the same dimensions"
        )
      end

      rank.times do |i|
        if i != axis && t.shape[i] != shape[i]
          raise Num::Internal::ShapeError.new(
            "All inputs must share a shape off-axis"
          )
        end
      end
      shape[axis] += t.shape[axis]
    end
    shape
  end
end
