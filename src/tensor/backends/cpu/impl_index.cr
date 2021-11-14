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
  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to which values will be assigned
  # * args : `Tuple` - Tuple of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor | Number` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
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
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to which values will be assigned
  # * args : `Array` - Array of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor(V, CPU(V))` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
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
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to which values will be assigned
  # * args : `Array` - Tuple of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `V` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
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

  # Return a shallow copy of a `Tensor` with a new dtype.  The underlying
  # data buffer is shared, but the `Tensor` owns its other attributes.
  # The size of the new dtype must be a multiple of the current dtype
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to view as a different data type
  # * u : `V.class` - The data type used to reintepret the underlying data buffer
  #   of a `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.view(Int8) # => [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
  # ```
  def view(arr : Tensor(U, CPU(U)), dtype : V.class) forall U, V
    s0 = sizeof(U)
    s1 = sizeof(V)

    shape = arr.shape.dup
    s0g = s0 > s1

    m = s0g ? (s0 // s1) : (s1 // s0)

    offset = s0g ? arr.offset * m : arr.offset // m
    shape[-1] = s0g ? shape[-1] * m : shape[-1] // m

    strides = Num::Internal.shape_to_strides(shape)
    data = arr.to_unsafe.unsafe_as(Pointer(V))
    storage = CPU(V).new(data, shape, strides)
    Tensor(V, CPU(V)).new(storage, shape, strides, offset, arr.flags.dup)
  end

  # Returns a view of the diagonal of a `Tensor`.  This method only works
  # for two-dimensional arrays.
  #
  # TODO: Implement views for offset diagonals
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to view along the diagonal
  #
  # ## Examples
  #
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
