require "./tensor"
require "./matrix"
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
    def {{name}}(x1 : Tensor, x2 : Tensor)
      if x1.size != x2.size
        raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
      end

      # TODO: Implement masking to use the *where* parameter
      Tensor.new(x1.size) do |i|
        x1[i] {{operator.id}} x2[i]
      end
    end

    def {{name}}(x1 : Matrix, x2 : Matrix)
      if Testing.matrix_aligned(x1, x2)
        return Matrix.new(x1.nrows, x1.nrows) do |i, j|
          x1[i, j] {{operator.id}} x2[i, j]
        end
      elsif Testing.broadcast_columns_first(x1, x2)
        return Matrix.new(x1.nrows, x2.ncols) do |i, j|
          x1[i, 0] {{operator.id}} x2[i, j]
        end
      elsif Testing.broadcast_columns_second(x1, x2)
        return Matrix.new(x1.nrows, x1.ncols) do |i, j|
          x1[i, j] {{operator.id}} x2[i, 0]
        end
      elsif Testing.broadcast_rows_first(x1, x2)
        return Matrix.new(x2.nrows, x1.ncols) do |i, j|
          x1[0, j] {{operator.id}} x2[i, j]
        end
      elsif Testing.broadcast_rows_second(x1, x2)
        return Matrix.new(x1.nrows, x1.ncols) do |i, j|
          x1[i, j] {{operator.id}} x2[0, j]
        end
      end
      raise "Matrix shapes are not aligned"
    end

    # {{name}}s two tensors with each other elementwise, storing
    # the result in *dest*
    #
    # ```
    # t1 = Tensor.new [1, 2, 3]
    # t2 = Tensor.empty(t1.size)
    #
    # B.{{name}}(t1, t1, dest: t2)
    # ```
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
    # ```
    # t1 = Tensor.new [1, 2, 3]
    # t2 = 5
    #
    # B.{{name}}(t1, t2)
    # ```
    def {{name}}(x1 : Tensor, x2 : Number, where : Tensor? = nil)
      # TODO: Implement masking to use the *where* parameter
      Tensor.new(x1.size) do |i|
        x1[i] {{operator.id}} x2
      end
    end

    # {{name}}s a matrix with a scalar elementwise
    #
    # ```
    # m = Matrix.new [[1, 2], [3, 4]]
    # B.{{name}}(m, 5)
    # ```
    def {{name}}(x1 : Matrix, x2 : Number)
      Matrix.new(x1.nrows, x2.ncols) do |i, j|
        x1[i, j] {{operator.id}} x2
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
      Tensor.new(x1.size) do |i|
        x2 {{operator.id}} x1[i]
      end
    end

    # {{name}}s a scalar with a matrix elementwise
    #
    # ```
    # m = Matrix.new [[1, 2], [3, 4]]
    # B.{{name}}(m, 5)
    # ```
    def {{name}}(x1 : Matrix, x2 : Number)
      Matrix.new(x1.nrows, x2.ncols) do |i, j|
        x2 {{operator.id}} x1[i, j]
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
