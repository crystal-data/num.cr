require "./tensor"

module Bottle::Statistics
  extend self

  # Computes the total sum of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # sum(v) # => 10
  # ```
  def sum(a : Tensor(U)) forall U
    a.flat_iter.reduce(U.new(0)) { |i, j| i + j.value }
  end

  def sum(a : Tensor, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      j.value += i.value
    end
  end

  # Computes the average of all Tensor values
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # mean(v) # => 2.5
  # ```
  def mean(a : Tensor)
    a.sum / a.size
  end

  def mean(a : Tensor, axis : Int32)
    n = a.shape[axis]
    a.reduce_along_axis(axis) do |i, j|
      j.value += i.value / n
    end
  end

  # Computes the standard deviation of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # std(v) # => 1.118
  # ```
  def std(a : Tensor)
    avg = mean(a)
    r = power(a - avg, 2)
    Math.sqrt(r.sum / a.size)
  end

  # Computes the median value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # median(v) # => 2.5
  # ```
  def median(a : Tensor)
    n = a.size
    sorted = a.sort
    if n % 2
      sorted[(n - 1) // 2]
    end
    m = (n - 1) / 2
    B.mean(sorted[[m.floor.to_i32, m.ceil.to_i32]])
  end

  # Computes the maximum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # max(v) # => 4
  # ```
  def max(a : Tensor(U)) forall U
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

  def max(a : Tensor, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      if i.value > j.value
        j.value = i.value
      end
    end
  end

  # Computes the minimum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # min(v) # => 1
  # ```
  def min(a : Tensor(U)) forall U
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

  def min(a : Tensor, axis : Int32)
    a.reduce_along_axis(axis) do |i, j|
      if i.value < j.value
        j.value = i.value
      end
    end
  end

  # Computes the "peak to peak" of a Tensor (max - min)
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp(v : Tensor)
    max(v) - min(v)
  end

  def ptp(a : Tensor, axis : Int32)
    max(a, axis) - min(a, axis)
  end
end
