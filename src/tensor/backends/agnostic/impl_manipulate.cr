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
  @[AlwaysInline]
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
  @[AlwaysInline]
  def concatenate(*arrs : Tensor(U, V), axis : Int) forall U, V
    concatenate(arrs.to_a, axis)
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
  # Num.vstack([a, a])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[AlwaysInline]
  def vstack(arrs : Array(Tensor(U, V))) forall U, V
    concatenate(arrs, 0)
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
  # Num.vstack([a, a])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  @[AlwaysInline]
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
  @[AlwaysInline]
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
  @[AlwaysInline]
  def hstack(*arrs : Tensor(U, V)) forall U, V
    concatenate(arrs.to_a, 1)
  end
end
