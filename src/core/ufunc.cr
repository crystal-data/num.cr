module Bottle::Internal::UFunc
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
    def {{name}}(x1 : BaseArray, x2 : BaseArray) forall U
      if x1.shape != x2.shape
        raise "Shapes {#{x1.shape}} and {#{x2.shape} are not aligned"
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
      x1.basetype.new(x2.shape) do |_|
        x1 {{operator.id}} ret.next.value
      end
    end

    # Returns the universal {{name}} function. Used to
    # apply outer operations, reductions, and accumulations
    # to tensors
    #
    # B.{{name}} # => <ufunc> {{name}}
    def {{name}}
      UFunc_{{name}}.new
    end

    # :nodoc:
    struct UFunc_{{name}}

      # A basic string representation of a
      # universal function.
      #
      # TODO: Add the same string representation
      # to the functions the struct contains
      def to_s(io)
        io << "<ufunc> {{ name }}"
      end

      # Applies an outer operations between two `Tensor`s.
      # Returns an MxN matrix where M is the size of *x1*,
      # and N is the size of *x2*
      #
      # ```
      # t = Tensor.new [1, 2]
      #
      # p B.add.outer(t, t)
      #
      # # Matrix[[  2  3]
      # #        [  3  4]]
      # ```
      def outer(x1 : BaseArray(U), x2 : BaseArray(V)) forall U, V
        outer = x1.unsafe_iter
        inner = x2.unsafe_iter
        c1 = uninitialized U
        c2 = uninitialized V
        x1.basetype.new(x1.shape + x2.shape) do |i|
          d = i % x2.size
          if d == 0
            c1 = outer.next.value
            inner = x2.unsafe_iter
          end
          c1 {{operator.id}} inner.next.value
        end
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
