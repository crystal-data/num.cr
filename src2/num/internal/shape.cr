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

module Num::Internal

  extend self
  # Converts a shape and a memory layout to valid strides for
  # that shape.
  #
  # ```
  # shape_to_strides([2, 2]) => [2, 1]
  # shape_to_strides([2, 2], Num::ColMajor) => [1, 2]
  # ```
  def shape_to_strides(
    shape : Array(Int32),
    layout : Num::OrderType = Num::RowMajor
  ) : Array(Int32)
    accum = 1
    ret = shape.clone
    case layout
    when Num::RowMajor
      (shape.size - 1).step(to: 0, by: -1) do |i|
        ret[i] = accum
        accum *= shape[i]
      end
    else
      shape.size.times do |i|
        ret[i] = accum
        accum *= shape[i]
      end
    end
    ret
  end

  # Calculates the N-dimensional shape of a standard
  # library `Array`.  All sub-arrays must be the same
  # length, and all elements of the (eventually)
  # flattened array must be of the same type.
  def stdlib_array_to_nd_shape(ary, shape : Array(Int32) = [] of Int32)
    return shape unless ary.is_a?(Array)
    r0 = ary[0]

    if r0.is_a?(Array)
      c0 = r0.size
      uniform = ary.all? do |row|
        row.is_a?(Array) && row.size == c0
      end
      unless uniform
        raise Num::Exceptions::ValueError.new("All subarrays must be the same length")
      end
    end

    shape << ary.size
    stdlib_array_to_nd_shape(ary[0], shape)
    shape
  end
end
