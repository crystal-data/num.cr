require "../libs/dtype"
require "../libs/cblas"
require "../flask/*"
require "../jug/*"

macro blas_helper(dtype, blas_prefix, cast)
  module LL
    extend self


    def matmul(a : Jug({{dtype}}), b : Jug({{dtype}}))
      c = Jug({{dtype}}).empty(a.nrows, b.ncols)
      LibCblas.{{blas_prefix}}gemm(
        LibCblas::MatrixLayout::RowMajor,
        LibCblas::MatrixTranspose::NoTrans,
        LibCblas::MatrixTranspose::NoTrans,
        a.nrows,
        b.ncols,
        a.ncols,
        1{{cast}},
        a.data,
        a.istride,
        b.data,
        b.istride,
        0{{cast}},
        c.data,
        c.istride,
      )
      return c
    end
  end
end

blas_helper Float64, d, _f64
blas_helper Float32, s, _f32
