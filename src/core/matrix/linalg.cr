require "../../matrix/*"
require "../../libs/gsl"
require "../../libs/lapack"
require "../../libs/cblas"

module Bottle::Core::Linalg
  extend self

  def matrix_inverse(matrix : Matrix(LibGsl::GslMatrix, Float64))
    sz = [matrix.ncols, matrix.nrows].min
    ipiv = Pointer(Int32).malloc(sz)
    info = 0
    m = matrix.nrows.to_i32
    n = matrix.ncols.to_i32
    lda = matrix.tda.to_i32

    LibLapack.dgetrf(
      pointerof(m),
      pointerof(n),
      matrix.data,
      pointerof(lda),
      ipiv,
      pointerof(info)
    )

    order = sz.to_i32
    lwork = m * n
    work = Pointer(Float64).malloc(lwork)

    LibLapack.dgetri(
      pointerof(order),
      matrix.data,
      pointerof(lda),
      ipiv,
      work,
      pointerof(lwork),
      pointerof(info),
    )
    return matrix
  end
end
