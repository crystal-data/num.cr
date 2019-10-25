require "../libs/dtype"
require "../libs/cblas"
require "../core/tensor"
require "../core/matrix"

module Bottle
  macro blas_helper(dtype, blas_prefix, cast)
    module Bottle::B
      extend self

      def outer(a : Tensor({{dtype}}), b : Tensor({{dtype}}))
        jug = Matrix.empty(a.size, b.size, dtype: {{dtype}})
        LibCblas.{{blas_prefix}}ger(
          LibCblas::MatrixLayout::RowMajor,
          a.size,
          b.size,
          1{{cast}},
          a.@buffer,
          a.@stride,
          b.@buffer,
          b.@stride,
          jug.@buffer,
          jug.@tda,
        )
        jug
      end

      def dot(a : Matrix({{dtype}}), x : Tensor({{dtype}}))
        flask = Tensor.empty(x.size, dtype: {{dtype}})
        LibCblas.{{blas_prefix}}gemv(
          LibCblas::MatrixLayout::RowMajor,
          LibCblas::MatrixTranspose::NoTrans,
          a.nrows,
          a.ncols,
          1{{cast}},
          a.@buffer,
          a.@tda,
          x.@buffer,
          x.@stride,
          0{{cast}},
          flask.@buffer,
          flask.@stride,
        )
        flask
      end
    end
  end

  blas_helper Float64, d, _f64
  blas_helper Float32, s, _f32
end
