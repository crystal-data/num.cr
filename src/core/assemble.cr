require "./ndtensor"

module Bottle::Internal::Assemble
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
  def concatenate(alist : Array(Tensor(U)), axis : Int32) forall U
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
    ret = Tensor(U).new(newshape)
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
  def vstack(alist : Array(Tensor(U))) forall U
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
  def hstack(alist : Array(Tensor(U))) forall U
    concatenate(alist, 1)
  end
end
