require "../core/tensor"
require "../core/matrix"

# Module for handling bitwise operations
# along Tensor's and Matrices.
#
# Only Boolean and Integer arrays are supported for
# these operations.
module Bottle::Internal::Binary
  macro binary_op(operator, name)

    # {{name}}s two tensors with each other elementwise
    #
    # t1 = Tensor.new [true, true, false]
    # t2 = Tensor.new [true, false, true]
    #
    # B.{{name}}(t1, t2)
    def {{name}}(x1 : Tensor(U), x2 : Tensor(U), where : Tensor(Bool)? = nil) forall U
      check_type(U)
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
    # t1 = Tensor.new [true, false, true]
    # t2 = Tensor.empty(t1.size)
    #
    # B.{{name}}(t1, t1, dest: t2)
    def {{name}}(x1 : Tensor(U), x2 : Tensor(U), dest : Tensor(U), where : Tensor(Bool)? = nil) forall U
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
    # t1 = Tensor.new [true, true, false]
    # t2 = 5
    #
    # B.{{name}}(t1, t2)
    def {{name}}(x1 : Tensor(U), x2 : Number, where : Tensor(Bool)? = nil) forall U
      # TODO: Implement masking to use the *where* parameter
      Tensor.new(x1.size) do |i|
        x1[i] {{operator.id}} x2
      end
    end

    # {{name}}s a scalar with a tensor elementwise.
    #
    # x = true
    # t = Tensor.new [true, false, true]
    #
    # B.{{name}}(x, t)
    def {{name}}(x1 : Number, x2 : Tensor(U), where : Tensor(Bool)? = nil) forall U
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
      # t = Tensor.new [true, false]
      #
      # p B.bitwise_and.outer(t, t)
      #
      # # Matrix[[   true  false]
      # #        [  false  false]]
      # ```
      def outer(x1 : Tensor(U), x2 : Tensor(U)) forall U
        Matrix.new(x1.size, x2.size) do |i, j|
          x1[i] {{operator.id}} x2[j]
        end
      end

      # Applies an accumulation function along
      # a `Tensor`.  Returns a copy of the `Tensor`
      #
      # ```
      # t = Tensor.new [true, true, true, false, false]
      #
      # t.bitwise_and.accumulate # => [true, true, true, false, false]
      # ```
      def accumulate(x1 : Tensor(U))
        ret = x1.clone
        (1...x1.size).each do |i|
          ret[i] = ret[i] {{ operator.id }} ret[i - 1]
        end
        ret
      end
    end
  end

  binary_op(:&, bitwise_and)
  binary_op(:|, bitwise_or)
  binary_op(:^, bitwise_xor)
  binary_op(:<<, left_shift)
  binary_op(:>>, right_shift)
end
