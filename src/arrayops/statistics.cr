require "../base/base"

module Bottle::Statistics
  extend self

  # Computes the total sum of a BaseArray
  #
  # ```
  # v = BaseArray.new [1, 2, 3, 4]
  # sum(v) # => 10
  # ```
  def sum(a : BaseArray(U)) forall U
    a.flat_iter.reduce(U.new(0)) { |i, j| i + j.value }
  end

  def sum(a : BaseArray, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      j.value += i.value
    end
  end

  def all(a : BaseArray(U)) forall U
    ret = a.astype(Bool)
    ret.flat_iter.reduce(true) { |i, j| i & j.value }
  end

  def all(a : BaseArray(U), axis : Int32) forall U
    ret = a.astype(Bool)
    ret.reduce_along_axis(axis) do |i, j|
      j.value &= i.value
    end
  end

  def any(a : BaseArray)
    ret = a.astype(Bool)
    ret.flat_iter.reduce(true) { |i, j| i | j.value }
  end

  def any(a : BaseArray(U), axis : Int32) forall U
    ret = a.astype(Bool)
    ret.reduce_along_axis(axis) do |i, j|
      j.value |= i.value
    end
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

  def mean(a : BaseArray, axis : Int32)
    n = a.shape[axis]
    a.reduce_along_axis(axis) do |i, j|
      j.value += i.value / n
    end
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

  def max(a : BaseArray, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      if i.value > j.value
        j.value = i.value
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

  def min(a : BaseArray, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      if i.value < j.value
        j.value = i.value
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
end
