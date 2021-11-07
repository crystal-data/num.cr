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
  # * arr : `Tensor(U, OCL(U))` - `Tensor` to which values will be assigned
  # * args : `Tuple` - Tuple of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor | Number` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # a[1.., 1..] = 99
  # a.cpu
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def set(arr : Tensor(U, OCL(U)), *args, value) forall U
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
  # * arr : `Tensor(U, OCL(U))` - `Tensor` to which values will be assigned
  # * args : `Array` - Array of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `Tensor(U, OCL(U))` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def set(arr : Tensor(U, OCL(U)), args : Array, t : Tensor(U, OCL(U))) forall U
    selected = arr[args]
    if t.rank > selected.rank
      raise Num::Exceptions::ValueError.new("Setting a Tensor with a sequence")
    end
    call_opencl_kernel(
      U,
      AssignmentKernel,
      [Int32, UInt32, Float32, Float64],
      selected, t
    )
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
  # * arr : `Tensor(U, OCL(U))` - `Tensor` to which values will be assigned
  # * args : `Array` - Array of arguments.  All arguments must be valid
  #   indexers, so a `Range`, `Int`, or `Tuple(Range, Int)`.
  # * value : `U` - Argument to assign to the `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def set(arr : Tensor(U, OCL(U)), args : Array, t : U) forall U
    selected = arr[args]
    call_opencl_kernel(
      U,
      AssignmentScalarKernel,
      [Int32, UInt32, Float32, Float64],
      selected, t
    )
  end
end
