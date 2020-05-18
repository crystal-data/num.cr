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

  def assert_shape_off_axis(ts, axis, shape)
    ts.each do |t|
      if t.shape.size != shape.size
        raise ShapeError.new("All inputs must share the same number of axes")
      end

      shape.size.times do |i|
        if i != axis && t.shape[i] != shape[i]
          raise ShapeError.new("All inputs must share a shape off axis")
        end
      end
      shape[axis] += t.shape[axis]
    end
    shape
  end

  def assert_shape(shape, ts)
    ts.each do |t|
      unless t.shape == shape
        raise ShapeError.new("All inputs must be the same shape")
      end
    end
  end

  def normalize_axis(axis, ndims)
    if axis < 0
      axis += ndims
    end
    if axis >= ndims || (axis < 0)
      raise ValueError.new("Axis out of range for array")
    end
    axis
  end

  def normalize_axies_list(axes, ndims)
    axes.map { |e| normalize_axis(e, ndims) }
  end

  def raise_zerod(items)
    if items.any? { |i| i.ndims == 0 }
      raise ShapeError.new("Zero dimensional arrays cannot be concatenated")
    end
  end

  def clip_axis(axis, size)
    if axis < 0
      axis += size
    end
    if axis < 0 || axis > size
      raise AxisError.new("Axis out of range")
    end
    axis
  end

  def less_than_3d(ts)
    if ts.any? { |t| t.ndims > 3 }
      raise NumInternal::ShapeError.new("Arrays must have less than 3 dimensions")
    end
  end

  def less_than_2d(ts)
    if ts.any? { |t| t.ndims > 2 }
      raise NumInternal::ShapeError.new("Arrays must have less than 3 dimensions")
    end
  end
end
