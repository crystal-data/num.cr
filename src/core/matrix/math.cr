require "../../matrix/*"

module Bottle::Core::MatrixMath
  include Bottle::Core::Exceptions
  extend self

  # Adds two matrices together elementwise
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a + a # => [[2, 4], []]
  # ```
  def add(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_add(a.ptr, b.ptr)
    return a
  end

  # Adds a constant to a matrix
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a + 2 # => [[3, 4], [5, 6]]
  # ```
  def add_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_add_constant(a.ptr, x)
    return a
  end

  # Subtracts two matrices together elementwise
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a - a # => [[0, 0], [0, 0]]
  # ```
  def sub(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_sub(a.ptr, b.ptr)
    return a
  end

  # Subtracts a constant from a matrix
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a - 2 # => [[-1, 0], [1, 2]]
  # ```
  def sub_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_add_constant(a.ptr, -x)
    return a
  end

  # Multiples two matrices together elementwise
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a * a # => [[1, 4], [9, 16]]
  # ```
  def mul_elements(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_mul_elements(a.ptr, b.ptr)
    return a
  end

  # Multiplies a matrix by a constant
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a * 2 # => [[2, 4], [6, 8]]
  # ```
  def mul_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_scale(a.ptr, x)
    return a
  end

  # Divides a matrix elementwise by another matrix
  #
  # ```
  # a = Matrix.new [[1, 2], [3, 4]]
  # a / a # => [[1, 1], [1, 1]]
  # ```
  def div_elements(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_div_elements(a.ptr, b.ptr)
    return a
  end

  # Divides a matrix by a constant
  #
  # ```
  # a = [[1, 2], [3, 4]]
  # a / 2 # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def div_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_sub(a.ptr, x)
    return a
  end
end
