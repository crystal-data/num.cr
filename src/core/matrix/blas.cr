require "../../matrix/*"
require "../../vector/*"
require "../../libs/cblas"
require "../../libs/gsl"

module Bottle::Core::MatrixBlas
  extend self

  # Multiplies a matrix times a vector
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # v = Vector.new [1, 2, 3]
  # m.mul(v) # => [14.0, 32.0, 50.0]
  # ```
  def mul_vector(a : Matrix(LibGsl::GslMatrix, Float64), x : Vector(LibGsl::GslVector, Float64))
    y = Vector.empty(x.size)
    LibCblas.dgemv(
      LibCblas::MatrixLayout::RowMajor,
      LibCblas::MatrixTranspose::NoTrans,
      a.nrows,
      a.ncols,
      1.0,
      a.data,
      a.tda,
      x.data,
      x.stride,
      0.0,
      y.data,
      y.stride,
    )
    return y
  end

  # Multiplies a matrix times a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.matmul(m) # => [[30.0, 36.0, 42.0], [66.0, 81.0, 96.0], [102.0, 126.0, 150.0]]
  # ```
  def mul_matrix(a : Matrix(LibGsl::GslMatrix, Float64), b : Matrix(LibGsl::GslMatrix, Float64))
    c = Matrix.empty(a.nrows, b.ncols)
    LibCblas.dgemm(
      LibCblas::MatrixLayout::RowMajor,
      LibCblas::MatrixTranspose::NoTrans, LibCblas::MatrixTranspose::NoTrans,
      a.nrows, b.ncols, a.ncols,
      1.0, a.data, a.tda,
      b.data, b.tda,
      1.0, c.data, c.tda,
    )
    return c
  end
end
