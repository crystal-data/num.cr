require "../libs/dtype"
require "../libs/cblas"
require "../core/flask"
require "../core/jug"

module Bottle
  macro blas_helper(dtype, blas_prefix, cast)
    module Bottle::LL
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
          a.tda,
          b.data,
          b.tda,
          0{{cast}},
          c.data,
          c.tda,
        )
        return c
      end
    end
  end

  blas_helper Float64, d, _f64
  blas_helper Float32, s, _f32
end
