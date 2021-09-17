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

  # Reduces a `Tensor` to a scalar by summing all of its
  # elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.sum(a) # => 6
  # ```
  @[AlwaysInline]
  def sum(a : Tensor(U, CPU(U))) forall U
    a.reduce { |i, j| i + j }
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  # *axis* : Int
  #   Axis of summation
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.sum(a, 0) # => [2, 4]
  # Num.sum(a, 1, dims: true)
  # # [[1],
  # #  [5]]
  # ```
  def sum(a : Tensor(U, CPU(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| i + j }
  end
end
