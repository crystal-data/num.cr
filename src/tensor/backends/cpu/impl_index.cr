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
  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def slice(arr : Tensor(U, CPU(U)), *args) forall U
    slice(arr, args.to_a)
  end

  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def slice(arr : Tensor(U, CPU(U)), args : Array) forall U
    new_shape = arr.shape.dup
    new_strides = arr.strides.dup

    acc = args.map_with_index do |arg, i|
      s_i, st_i, o_i = normalize(arg, i)
      new_shape[i] = s_i
      new_strides[i] = st_i
      o_i
    end

    i = 0
    new_strides.reject! do
      condition = new_shape[i] == 0
      i += 1
      condition
    end

    new_shape.reject! do |j|
      j == 0
    end

    offset = arr.offset

    rank.times do |k|
      if arr.strides[k] < 0
        offset += (arr.shape[k] - 1) * arr.strides[k].abs
      end
    end

    acc.zip(self.strides) do |a, j|
      offset += a * j
    end

    Tensor.new(arr.data, new_shape, new_strides, offset, U)
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # Arguments
  # ---------
  # *args* : *U
  #   Tuple of arguments.  All but the last argument must be valid
  #   indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.  The final
  #   argument passed is used to set the values of the `Tensor`.  It can
  #   be either a `Tensor`, or a scalar value.
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def assign(arr : Tensor(U, CPU(U)), *args : *V) forall U, V
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # Arguments
  # ---------
  # *args* : *U
  #   Tuple of arguments.  All but the last argument must be valid
  #   indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.  The final
  #   argument passed is used to set the values of the `Tensor`.  It can
  #   be either a `Tensor`, or a scalar value.
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def assign(arr : Tensor(U, CPU(U)), args : Array, value) forall U, V
  end

  # Return a shallow copy of a `Tensor`.  The underlying data buffer
  # is shared, but the `Tensor` owns its other attributes.  Changes
  # to a view of a `Tensor` will be reflected in the original `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).new([3, 3])
  # b = a.view
  # b[...] = 99
  # a
  #
  # # [[99, 99, 99],
  # #  [99, 99, 99],
  # #  [99, 99, 99]]
  # ```
  def view(arr : Tensor(U, CPU(U))) forall U
  end

  # Return a shallow copy of a `Tensor` with a new dtype.  The underlying
  # data buffer is shared, but the `Tensor` owns its other attributes.
  # The size of the new dtype must be a multiple of the current dtype
  #
  # Arguments
  # ---------
  # *u* : U.class
  #   The data type used to reintepret the underlying data buffer
  #   of a `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.view(Int16) # => [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
  # ```
  def view(arr : Tensor(U, CPU(U)), dtype : U.class) forall U
  end

  # Returns a view of the diagonal of a `Tensor`.  This method only works
  # for two-dimensional arrays.
  #
  # TODO: Implement views for offset diagonals
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(3, 3) { |i, _| i }
  # a.diagonal # => [0, 1, 2]
  # ```
  def diagonal(arr : Tensor(U, CPU(U)))
  end

  private def normalize(arr : Tensor, arg : Int, i : Int32)
    if arg < 0
      arg += arr.shape[i]
    end
    if arg < 0 || arg >= arr.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{arr.shape[i]}"
    end
    {0, 0, arg.to_i}
  end

  private def normalize(arr, arg : Range, i : Int32)
    a_end = arg.end
    if a_end.is_a?(Int32)
      if a_end > self.shape[i]
        arg = arg.begin...self.shape[i]
      end
    end
    s, o = Indexable.range_to_index_and_count(arg, self.shape[i])
    if s >= self.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{self.shape[i]}"
    end
    {o.to_i, self.strides[i], s.to_i}
  end

  private def normalize(arg : Tuple(Range(B, E), Int), i : Int32) forall B, E
    range, step = arg
    abs_step = step.abs
    start, offset = Indexable.range_to_index_and_count(range, self.shape[i])
    if start >= self.shape[i]
      raise "Index #{arg} out of range for axis #{i} with size #{self.shape[i]}"
    end
    {offset // abs_step + offset % abs_step, step * self.strides[i], start}
  end
end
