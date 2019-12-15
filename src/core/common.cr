require "../base/base"
require "./exceptions"

module Num::Internal
  include Exceptions

  private def assert_shape_off_axis(ts, axis, shape)
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

  private def assert_shape(shape, ts)
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
    axes.map { |e| normalize_axis(axis, ndims) }
  end
end
