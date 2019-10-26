require "../core/tensor"
require "../core/matrix"

module Bottle
  macro ufunc(operator, name)
    module B
      extend self
      # {{name}}s two tensors with each other elementwise
      #
      # t1 = Tensor.new [1, 2, 3]
      # t2 = Tensor.new [4, 5, 6]
      #
      # B.{{name}}(t1, t2)
      def {{name}}(x1 : Tensor, x2 : Tensor, where : Tensor? = nil)
        if x1.size != x2.size
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end

        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.size) do |i|
          x1[i] {{operator.id}} x2[i]
        end
      end

      # {{name}}s two tensors with each other elementwise, storing
      # the result in *dest*
      #
      # t1 = Tensor.new [1, 2, 3]
      # t2 = Tensor.empty(t1.size)
      #
      # B.{{name}}(t1, t1, dest: t2)
      def {{name}}(x1 : Tensor, x2 : Tensor, dest : Tensor, where : Tensor? = nil)
        if x1.size != x2.size
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end

        # TODO: Implement masking to use the *where* parameter
        x1.size.times do |i|
          dest[i] = x1[i] {{ operator.id }} x2[i]
        end
      end

      # {{name}}s a tensor with a scalar elementwise.
      #
      # t1 = Tensor.new [1, 2, 3]
      # t2 = 5
      #
      # B.{{name}}(t1, t2)
      def {{name}}(x1 : Tensor, x2 : Number, where : Tensor? = nil)
        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.size) do |i|
          x1[i] {{operator.id}} x2
        end
      end

      # {{name}}s a scalar with a tensor elementwise.
      #
      # x = 5
      # t = Tensor.new [1, 2, 3]
      #
      # B.{{name}}(x, t)
      def {{name}}(x1 : Number, x2 : Tensor, where : Tensor? = nil)
        {{name}}(x2, x1, where)
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
        def outer(x1 : Tensor, x2 : Tensor)
          Matrix.new(x1.size, x2.size) do |i, j|
            x1[i] {{operator.id}} x2[j]
          end
        end

        # Applies an accumulation function along
        # a `Tensor`.  Returns a copy of the `Tensor`
        #
        # ```
        # t = Tensor.new [1, 2, 3, 4, 5]
        #
        # t.add.accumulate # => [1, 3, 6, 10, 15]
        # ```
        def accumulate(x1 : Tensor)
          ret = x1.clone
          (1...x1.size).each do |i|
            ret[i] = ret[i] {{ operator.id }} ret[i - 1]
          end
          ret
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
  ufunc(:&, bitwise_and)
  ufunc(:|, bitwise_or)
  ufunc(:^, bitwise_xor)
  ufunc(:==, equal)
  ufunc(:>, greater)
  ufunc(:>=, greater_equal)
  ufunc(:<, less)
  ufunc(:<=, less_equal)
end
