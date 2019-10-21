require "../libs/lapack"
require "../core/jug"
require "../core/flask"

module Bottle
  macro linalg_helper(dtype, blas_prefix)
    module LA
      extend self
      def inv(a : Jug({{dtype}}))
        dim = Math.min(a.nrows, a.ncols)
        ipiv = Pointer(Int32).malloc(dim)

        m = a.nrows
        n = a.ncols
        lda = a.tda

        LibLapack.{{blas_prefix}}getrf(
          pointerof(m),
          pointerof(n),
          a.data,
          pointerof(lda),
          ipiv,
          out info
        )

        order = dim
        lwork = m * n
        work = Pointer({{dtype}}).malloc(lwork)

        LibLapack.{{blas_prefix}}getri(
          pointerof(order),
          a.data,
          pointerof(lda),
          ipiv,
          work,
          pointerof(lwork),
          out invinfo,
        )

        return a
      end
    end
  end

  linalg_helper Float64, d
  linalg_helper Float32, s
end
