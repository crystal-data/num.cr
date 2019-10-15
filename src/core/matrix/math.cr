require "../../matrix/*"

module Bottle::Core::MatrixMath
  include Bottle::Core::Exceptions
  extend self

  def add(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_add(a.ptr, b.ptr)
    return a
  end

  def add_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_add_constant(a.ptr, x)
    return a
  end

  def sub(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_sub(a.ptr, b.ptr)
    return a
  end

  def sub_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_add_constant(a.ptr, -x)
    return a
  end

  def mul_elements(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_mul_elements(a.ptr, b.ptr)
    return a
  end

  def mul_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_scale(a.ptr, x)
    return a
  end

  def div_elements(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    LibGsl.gsl_matrix_div_elements(a.ptr, b.ptr)
    return a
  end

  def div_constant(a : Matrix(LibGsl::GslMatrix, Float64), x : Number)
    LibGsl.gsl_matrix_sub(a.ptr, x)
    return a
  end
end
