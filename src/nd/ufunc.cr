require "./ndtensor"
require "../util/testing"

module Bottle::NDimensional::UFunc
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
    def {{name}}(x1 : NDTensor, x2 : NDTensor)
      if x1.shape != x2.shape
        raise "Shapes {#{x1.shape}} and {#{x2.shape} are not aligned"
      end

      p1, s1 = x1.unsafe_flat_iter
      p2, s2 = x2.unsafe_flat_iter
      NDTensor.new(x1.shape.dims) do |i|
        p1[i * s1] {{operator.id}} p2[i * s2]
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
    def {{name}}(x1 : NDTensor, x2 : Number)
      ptr, s1 = x1.unsafe_flat_iter
      NDTensor.new(x1.shape.dims) do |i|
        ptr[i * s1] {{operator.id}} x2
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
    def {{name}}(x1 : Number, x2 : NDTensor)
      ptr, s1 = x2.unsafe_flat_iter
      NDTensor.new(x1.shape.dims) do |i|
        x2 {{operator.id}} ptr[i * s1]
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
      def outer(x1 : NDTensor, x2 : NDTensor)
        newshape = Shape.new(x1.shape.dims + x2.shape.dims)

        p1, s1 = x1.unsafe_flat_iter
        p2, s2 = x2.unsafe_flat_iter
        offset = 0

        NDTensor.new(x1.shape, x2.shape) do |i, j|
          p1[i] {{operator.id}} p2[j]
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
