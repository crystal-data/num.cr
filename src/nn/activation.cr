require "../tensor/tensor"
require "../tensor/linalg"
require "../num/math"

module Num
  def sigmoid(z : Tensor)
    z.map! do |i|
      1.0 / (1.0 + Num.exp(-i))
    end
  end

  def relu(z : Tensor)
    Num.max(0, z)
  end

  def tanh_nn(z : Tensor)
    Num.tanh(z)
  end

  def elu(z : Tensor)
    z.map do |el|
      el > 0 ? el : Num.exp(el) - 1
    end
  end
end
