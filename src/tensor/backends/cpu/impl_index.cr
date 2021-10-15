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
    offset, shape, strides = Num::Internal.offset_for_index(arr, args)
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    Tensor.new(arr.data, shape, strides, offset, flags, U)
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
  def set(arr : Tensor(U, CPU(U)), *args, value) forall U
    set(arr, args.to_a, value)
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
  def set(arr : Tensor(U, CPU(U)), args : Array, t : Tensor(V, CPU(V))) forall U, V
    s = arr[args]
    t = t.broadcast_to(s.shape)
    if t.rank > s.rank
      raise "Setting a Tensor with a sequence"
    end
    s.map!(t) do |_, j|
      j
    end
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
  def set(arr : Tensor(U, CPU(U)), args : Array, t : V) forall U, V
    s = arr[args]
    s.map! do
      t
    end
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
  def diagonal(arr : Tensor(U, CPU(U))) forall U
    unless arr.rank == 2
      raise Num::Exceptions::ValueError.new("Tensor must be 2D")
    end

    n = arr.shape.min
    new_shape = [n]
    new_strides = [arr.strides.sum]
    Tensor.new(arr.data, new_shape, new_strides, arr.offset, U)
  end
end
