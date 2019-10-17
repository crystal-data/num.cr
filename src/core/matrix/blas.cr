require "../../matrix/*"
require "../../vector/*"
require "../../libs/cblas"
require "../../libs/gsl"

module Bottle::Core::MatrixBlas
  extend self

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
