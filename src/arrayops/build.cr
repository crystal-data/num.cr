require "../base/base"

module Bottle::Assemble
  extend self

  # Concatenates an array of `Tensor's` along a provided axis.
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be concatenated
  # axis : Int32
  #   - Axis for concatentation, must be an existing
  #     axis present in all Tensors, and all Tensors
  #     must have the same shape off-axis
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  #
  # concatenate([t, t, t], axis=-1)
  #
  # Tensor([[[ 0,  1,  2,  0,  1,  2,  0,  1,  2],
  #          [ 3,  4,  5,  3,  4,  5,  3,  4,  5]],
  #
  #         [[ 6,  7,  8,  6,  7,  8,  6,  7,  8],
  #          [ 9, 10, 11,  9, 10, 11,  9, 10, 11]]])
  # ```
  def concatenate(alist : Array(BaseArray(U)), axis : Int32) forall U
    newshape = alist[0].shape.dup

    if axis < 0
      axis += newshape.size
    end

    if axis < 0 || axis > newshape.size
      raise "Axis out of range"
    end

    newshape[axis] = 0
    alist.each do |v|
      if (v.shape.size != newshape.size)
        raise "All inputs must have the same number of axes"
      end
      newshape.size.times do |i|
        if (i != axis && v.shape[i] != newshape[i])
          raise "All inputs must have the same shape off-axis"
        end
      end
      newshape[axis] += v.shape[axis]
    end
    ret = alist[0].class.new(newshape)
    lo = [0] * newshape.size
    hi = newshape.dup
    hi[axis] = 0
    alist.each do |v|
      if (v.shape[axis] != 0)
        hi[axis] += v.shape[axis]
        ranges = lo.zip(hi).map { |i, j| i...j }
        ret[ranges] = v
        lo[axis] = hi[axis]
      end
    end
    ret
  end

  # Concatenates a list of `Tensor`s along axis 0
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # vstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def vstack(alist : Array(BaseArray(U))) forall U
    concatenate(alist, 0)
  end

  # Concatenates a list of `Tensor`s along axis 1
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # hstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def hstack(alist : Array(BaseArray(U))) forall U
    concatenate(alist, 1)
  end

  def dstack(alist : Array(BaseArray(U))) forall U
    shape0 = alist[0].shape
    if !alist.all? { |t| t.shape == shape0 }
      raise Exceptions::ShapeError.new("All arrays must be the same shape")
    end

    if alist[0].ndims == 1
      alist = alist.map { |t| t.reshape([1, t.size, 1]) }
      concatenate(alist, 2)
    elsif alist[0].ndims == 2
      alist = alist.map { |t| t.reshape(t.shape + [1]) }
      concatenate(alist, 2)
    else
      raise Exceptions::ShapeError.new(
        "dstack only supports 1 and 2-dimensional arrays")
    end
  end

  def column_stack(alist : Array(BaseArray(U))) forall U
    shape0 = alist[0].shape
    if !alist.all? { |t| t.shape == shape0 }
      raise Exceptions::ShapeError.new("All arrays must be the same shape")
    end

    if alist[0].ndims == 1
      alist = alist.map { |t| t.reshape([t.size, 1]) }
      concatenate(alist, 1)
    elsif alist[0].ndims == 2
      alist = alist.map { |t| t.reshape(t.shape) }
      concatenate(alist, 1)
    else
      raise Exceptions::ShapeError.new(
        "column_stack only supports 1 and 2-dimensional arrays")
    end
  end
end
