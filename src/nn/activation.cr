require "../api"
require "../autograd/common"

module Num::NN
  def sigmoid_fn(z : Tensor(U)) forall U
    z.map do |i|
      U.new(1 / (1 + Num.exp(-i)))
    end
  end

  def sigmoid_backward(gradient : Tensor(U), cached : Tensor(U)) forall U
    cached_tensor.map2(gradient) do |x, y|
      x * (1 - x) * y
    end
  end

  class SigmoidActivation(T) < Gate(T)
    @cache : T

    def initialize(@cache)
    end

    def backward(payload : Payload(T))
      gradient = payload.variable.grad
      [sigmoid_backward(gradient, @cache)]
    end
  end

  class Variable(T)
    def sigmoid_cache(a : Variable(T))
      gate = SigmoidActivation(T).new(@value)

      @grad = @value.zeros_like
      @requires_grad = true
      register_node(
        "Sigmoid",
        gate,
        self,
        a
      )
    end

    def sigmoid
      result = Context(T).new(@context, sigmoid_fn(@value))

      if is_grad_needed
        result.sigmoid_cache(self)
      end
      result
    end
  end
end
