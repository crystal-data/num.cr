module Strides
  extend self

  def offset(i, strides_i, j = 0, strides_j = 0)
    (i * strides_i) + (j * strides_j)
  end

  def offset_row(i, strides_i, j, j_stride)
    return offset(i, strides_i), offset(j, j_stride)
  end
end
