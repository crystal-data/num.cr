require "./ndtensor"
require "../util/testing"

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
    def {{name}}(x1 : NDTensor, x2 : NDTensor)
      if x1.shape != x2.shape
        raise "Shapes {#{x1.shape}} and {#{x2.shape} are not aligned"
      end

      s1 = x1.strides[-1]
      s2 = x2.strides[-1]
      p1 = x1.@ptr
      p2 = x2.@ptr
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
    def {{name}}(x1 : Tensor, x2 : Number)
      s1 = x1.strides[-1]
      ptr = x1.@ptr
      NDTensor.new(x2.shape.dims) do |i|
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
    def {{name}}(x1 : Number, x2 : Tensor, where : Tensor? = nil)
      s1 = x2.strides[-1]
      ptr = x2.@ptr
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
      def outer(x1 : NDTensor(U), x2 : NDTensor(U)) forall U
        newshape = Shape.new(x1.shape.dims + x2.shape.dims)

        s1 = x1.strides[-1]
        s2 = x2.strides[-1]
        p1 = x1.@ptr
        p2 = x2.@ptr
        offset = 0

        ptr = Pointer(U).malloc(newshape.totalsize)

        x1.shape.totalsize.times do |i|
          x2.shape.totalsize.times do |j|
            ptr[offset] = p1[i] {{operator.id}} p2[j]
            offset += 1
          end
        end

        NDTensor.new(ptr, newshape)
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
