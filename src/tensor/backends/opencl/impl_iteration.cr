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
  extend self

  # Yields a view of each lane of an `axis`.  Changes made in
  # the passed block will be reflected in the original `Tensor`
  #
  # ## Arguments
  #
  # * a0 : `Tensor(U, OCL(U))` - `Tensor` to iterate along
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicates if the axis of reduction should be removed
  #   from the result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([3, 3], device: OCL) { |i| i }
  # a.each_axis(1) do |ax|
  #   puts ax.cpu
  # end
  #
  # # [0, 3, 6]
  # # [1, 4, 7]
  # # [2, 5, 8]
  # ```
  @[Inline]
  def each_axis(
    a0 : Tensor(U, OCL(U)),
    axis : Int,
    dims : Bool = false,
    &block : Tensor(U, OCL(U)) -> _
  ) forall U
    axis = normalize_axis_index(axis, a0.rank)
    0.step(to: a0.shape[axis] - 1) do |i|
      yield at_axis_index(a0, axis, i, dims)
    end
  end

  # Reduces a `Tensor` along an axis. Returns a `Tensor`, with the axis
  # of reduction either removed, or reduced to 1 if `dims` is True, which
  # allows the result to broadcast against its previous shape
  #
  # ## Arguments
  #
  # * a0 : `Tensor(U, OCL(U))` - `Tensor` to reduce along an axis
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Flag determining whether the axis of reduction should be
  #   kept in the result
  # * block : `Proc(Tensor(U, OCL(U)), Tensor(U, OCL(U)), _)` - `Proc` to
  #   apply to values along an axis
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.reduce_axis(0) { |i, j| Num.add!(i, j) } # => "<2> on OpenCL Backend"
  # ```
  @[Inline]
  def reduce_axis(
    a0 : Tensor(U, OCL(U)),
    axis : Int,
    dims : Bool = false,
    &block : Tensor(U, OCL(U)), Tensor(U, OCL(U)) -> _
  ) : Tensor(U, OCL(U)) forall U
    axis = normalize_axis_index(axis, a0.rank)
    memo = at_axis_index(a0, axis, 0, dims)
    result = Tensor(U, OCL(U)).zeros(memo.shape)
    yield result, memo
    1.step(to: a0.shape[axis] - 1) do |i|
      yield result, at_axis_index(a0, axis, i, dims)
    end
    result
  end
end
