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

require "./constants"

module NumInternal
  extend self

  # Takes in a shape, and a memory order type, and returns the
  # strides of an array matching the shape and layout.
  # Num.cr uses row major memory layout by default, and ClTensors
  # have this enforced, it is not possible to use col major storage
  # on a ClTensor.
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
end
