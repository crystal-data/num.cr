require "./ndtensor"

module Bottle::Internal::Assemble
  extend self

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

  def vstack(alist : Array(Tensor(U))) forall U
    concatenate(alist, 0)
  end

  def hstack(alist : Array(Tensor(U))) forall U
    concatenate(alist, 1)
  end
end
