require "../base/base"
require "../util/testing"

module Bottle::Binary
  extend self

  macro binary_op(operator, name)
    # {{name}}s two tensors with each other elementwise
    #
    # ```
    # t1 = Tensor.new [1, 2, 3]
    # t2 = Tensor.new [4, 5, 6]
    #
    # B.{{name}}(t1, t2)
    # ```
    def {{name}}(x1 : BaseArray, x2 : BaseArray)
      newshape = x1.broadcastable(x2)
      if newshape.size != 0
        x1 = x1.broadcast_to(newshape)
        x2 = x2.broadcast_to(newshape)
      end
      i1 = x1.unsafe_iter
      i2 = x2.unsafe_iter
      x1.basetype.new(x1.shape) do |_|
        i1.next.value {{operator.id}} i2.next.value
      end
    end

    # {{name}}s a tensor with a scalar elementwise.
    #
    # ```
    # t1 = Tensor.new [1, 2, 3]
    # t2 = 5
    #
    # B.{{name}}(t1, t2)
    # ```
    def {{name}}(x1 : BaseArray, x2 : Number)
      ret = x1.unsafe_iter
      x1.basetype.new(x1.shape) do |_|
        ret.next.value {{operator.id}} x2
      end
    end

    # {{name}}s a scalar with a tensor elementwise.
    #
    # ```
    # x = 5
    # t = Tensor.new [1, 2, 3]
    #
    # B.{{name}}(x, t)
    # ```
    def {{name}}(x1 : Number, x2 : BaseArray)
      ret = x2.unsafe_iter
      x2.basetype.new(x2.shape) do |_|
        x1 {{operator.id}} ret.next.value
      end
    end
  end

  binary_op(:&, bitwise_and)
  binary_op(:|, bitwise_or)
  binary_op(:^, bitwise_xor)
  binary_op(:<<, left_shift)
  binary_op(:>>, right_shift)
end
