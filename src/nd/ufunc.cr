require "./ndtensor"

module NDArray::Internal::UFunc
  extend self

  macro ufunc(operator, name)
    # {{name}}s two tensors with each other elementwise
    #
    # ```
    # t1 = Tensor.new [1, 2, 3]
    # t2 = Tensor.new [4, 5, 6]
    #
    # B.{{name}}(t1, t2)
    # ```
    def {{name}}(x1 : Tensor, x2 : Tensor)
      if x1.shape != x2.shape
        raise "Shapes {#{x1.shape}} and {#{x2.shape} are not aligned"
      end
      i1 = x1.unsafe_iter
      i2 = x2.unsafe_iter
      Tensor.new(x1.shape) do |_|
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
    def {{name}}(x1 : Tensor, x2 : Number)
      ret = x1.unsafe_iter
      Tensor.new(x1.shape) do |_|
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
    def {{name}}(x1 : Number, x2 : Tensor)
      ret = x2.unsafe_iter
      Tensor.new(x2.shape) do |_|
        x1 {{operator.id}} ret.next.value
      end
    end
  end

  ufunc(:+, add)
  ufunc(:-, subtract)
  ufunc(:*, multiply)
  ufunc(:/, divide)
  ufunc(:**, power)
  ufunc(://, floordiv)
  ufunc(:%, modulo)
  ufunc(:==, equal)
  ufunc(:>, greater)
  ufunc(:>=, greater_equal)
  ufunc(:<, less)
  ufunc(:<=, less_equal)
end
