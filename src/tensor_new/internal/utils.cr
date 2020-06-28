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
#
require "./exceptions"

module Num::Internal
  extend self

  # :nodoc:
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

  # :nodoc:
  def calculate_array_shape(arr, calc_shape : Array(Int32) = [] of Int32)
    return calc_shape unless arr.is_a?(Array)
    first_el = arr[0]
    if first_el.is_a?(Array)
      lc = first_el.size
      unless arr.all? { |el| el.is_a?(Array) && el.size == lc }
        raise Num::Internal::ShapeError.new("All subarrays must be the same length")
      end
    end
    calc_shape << arr.size
    calc_shape = calculate_array_shape(arr[0], calc_shape)
    calc_shape
  end
end
