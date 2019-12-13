require "./macros"

module Num::Statistics
  include Internal
  extend self

  def sum(a : Tensor(U)) forall U
    reducescalar :+, 0, a
  end

  def sum(a : Tensor, axis : Int32, keepdims = false)
    reduceaxis :+, a
  end

  def prod(a : Tensor(U)) forall U
    reducescalar :*, 1, a
  end

  def prod(a : Tensor, axis : Int32, keepdims = false)
    reduceaxis :*, a
  end

  def all(a : BaseArray(U)) forall U
    ret = a.astype(Bool)
    reducebool :&, true, ret
  end

  def all(a : BaseArray(U), axis : Int32, keepdims = false) forall U
    ret = a.astype(Bool)
    reduceaxis :&, ret
  end

  def any(a : BaseArray)
    ret = a.astype(Bool)
    reducebool :|, true, ret
  end

  def any(a : BaseArray(U), axis : Int32, keepdims = false) forall U
    ret = a.astype(Bool)
    reduceaxis :|, ret
  end

  def sort(a : BaseArray(U), axis : Int32) forall U
    ret = a.dup
    ret.permute_along_axis(axis) do |perm|
      ii = perm.dup.to_unsafe.to_slice(perm.size).sort
      perm[...] = a.basetype.new(perm.shape) { |i| ii[i] }
    end
    ret
  end

  # Computes the average of all BaseArray values
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # mean(v) # => 2.5
  # ```
  def mean(a : BaseArray)
    a.sum / a.size
  end

  def mean(a : BaseArray(U), axis : Int32, keepdims = false) forall U
    sum(a, axis, keepdims) / a.shape[axis]
  end

  # Computes the standard deviation of a BaseArray
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # std(v) # => 1.118
  # ```
  def std(a : BaseArray)
    avg = mean(a)
    r = power(a - avg, 2)
    Math.sqrt(r.sum / a.size)
  end

  def std(a : BaseArray, axis : Int32, keepdims = false)
    avg = mean(a, axis, keepdims: true)
    r = power(a - avg, 2)
    N.sqrt(N.sum(r, axis, keepdims) / r.shape[axis])
  end

  # Computes the maximum value of a BaseArray
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # max(v) # => 4
  # ```
  def max(a : BaseArray(U)) forall U
    mx = uninitialized U
    a.flat_iter.each_with_index do |el, i|
      c = el.value
      if i == 0
        mx = c
      end
      if c > mx
        mx = c
      end
    end
    mx
  end

  def max(a : BaseArray, axis : Int32, keepdims = false)
    a.reduce_fast(axis, keepdims) do |i, j|
      if j.value > i.value
        i.value = j.value
      end
    end
  end

  # Computes the minimum value of a BaseArray
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # min(v) # => 1
  # ```
  def min(a : BaseArray(U)) forall U
    mx = uninitialized U
    a.flat_iter.each_with_index do |el, i|
      c = el.value
      if i == 0
        mx = c
      end
      if c < mx
        mx = c
      end
    end
    mx
  end

  def min(a : BaseArray, axis : Int32, keepdims = false)
    a.reduce_fast(axis, keepdims) do |i, j|
      if j.value < i.value
        i.value = j.value
      end
    end
  end

  # Computes the "peak to peak" of a BaseArray (max - min)
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp(v : BaseArray)
    max(v) - min(v)
  end

  def ptp(a : BaseArray, axis : Int32)
    max(a, axis) - min(a, axis)
  end

  def clip(a : BaseArray(U), min : Number, max : Number) forall U
    a = a.dup
    a.flat_iter.each do |i|
      if i.value < min
        i.value = U.new(min)
      elsif i.value > max
        i.value = U.new(max)
      end
    end
    a
  end
end
