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
  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, OCL(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], device: OCL) { |i| i }
  # Num.sum(a, 0).cpu # => [2, 4]
  # Num.sum(a, 1, dims: true).cpu
  # # [[1],
  # #  [5]]
  # ```
  @[AlwaysInline]
  def sum(a : Tensor(U, OCL(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| Num.add!(i, j) }
  end
end
