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

module Num::Internal
  extend self

  # :nodoc:
  #
  # Converts a shape to the appropriate strides for a particular memory
  # layout, be it RowMajor or ColMajor.  Will always return
  # contiguous strides
  #
  # ```
  # a = [3, 2, 2]
  # shape_to_strides(a, Num::RowMajor) # => [4, 2, 1]
  # ```
  def shape_to_strides(shape : Array(Int32), layout : Num::OrderType = Num::RowMajor) : Array(Int32)
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

  # :nodoc:
  #
  # Finds the N-dimensional shape of a stdlib Array, recursively.  Useful
  # for converting a deeply nested Array to a Tensor cleanly without
  # type confusion.
  def recursive_array_shape(a, shape : Array(Int32) = [] of Int32) : Array(Int32)
    unless a.is_a?(Array)
      return shape
    end

    e0 = a[0]

    if e0.is_a?(Array)
      s0 = e0.size
      uniform = a.all? do |el|
        el.is_a?(Array) && el.size == s0
      end

      raise "All subarrays must be the same length" unless uniform
    end

    shape << a.size
    shape = recursive_array_shape(a[0], shape)
    shape
  end
end
