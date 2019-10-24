require "../libs/dtype"
require "../libs/cblas"
require "../core/vector"
require "../core/matrix"

module Bottle
  macro blas_helper(dtype, blas_prefix, cast)
    module Bottle::B
      extend self

      def outer(a : Vector({{dtype}}), b : Vector({{dtype}}))
        jug = Matrix.empty(a.size, b.size, dtype: {{dtype}})
        LibCblas.{{blas_prefix}}ger(
          LibCblas::MatrixLayout::RowMajor,
          a.size,
          b.size,
          1{{cast}},
          a.data,
          a.stride,
          b.data,
          b.stride,
          jug.data,
          jug.tda,
        )
        jug
      end

      def dot(a : Matrix({{dtype}}), x : Vector({{dtype}}))
        flask = Flask.empty(x.size, dtype: {{dtype}})
        LibCblas.{{blas_prefix}}gemv(
          LibCblas::MatrixLayout::RowMajor,
          LibCblas::MatrixTranspose::NoTrans,
          a.nrows,
          a.ncols,
          1{{cast}},
          a.data,
          a.tda,
          x.data,
          x.stride,
          0{{cast}},
          flask.data,
          flask.stride,
        )
        flask
      end
    end
  end

  blas_helper Float64, d, _f64
  blas_helper Float32, s, _f32
end
