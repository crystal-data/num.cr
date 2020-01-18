require "../tensor/tensor"

module Num
  extend self

  def isfinite(x : Tensor(Float32) | Tensor(Float64))
    iter = x.unsafe_iter
    Tensor.new(x.shape) do |_|
      iter.next.value.finite?
    end
  end

  def isfinite(x : Tensor)
    Tensor.new(x.shape) { |_| true }
  end

  def isinf(x : Tensor(Float32) | Tensor(Float64))
    iter = x.unsafe_iter
    Tensor.new(x.shape) do |_|
      !!iter.next.value.infinite?
    end
  end

  def isinf(x : Tensor)
    Tensor.new(x.shape) { |_| false }
  end

  def isnan(x : Tensor(Float32) | Tensor(Float64))
    iter = x.unsafe_iter
    Tensor.new(x.shape) do |_|
      iter.next.value.nan?
    end
  end

  def isnan(x : Tensor)
    Tensor.new(x.shape) { |_| false }
  end

  def isneginf(x : Tensor(Float32) | Tensor(Float64))
    iter = x.unsafe_iter
    Tensor.new(x.shape) do |_|
      val = iter.next.value.infinite?
      val == -1
    end
  end

  def isneginf(x : Tensor)
    Tensor.new(x.shape) { |_| false }
  end

  def isposinf(x : Tensor(Float32) | Tensor(Float64))
    iter = x.unsafe_iter
    Tensor.new(x.shape) do |_|
      val = iter.next.value.infinite?
      val == 1
    end
  end

  def isposinf(x : Tensor)
    Tensor.new(x.shape) { |_| false }
  end
end
