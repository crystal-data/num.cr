require "../libs/lapack"
require "../core/tensor"
require "../core/matrix"

module Bottle
  macro linalg_helper(dtype, blas_prefix)
    module B::LinAlg
      extend self

      def inv(a : Matrix({{dtype}}))
        if a.nrows != a.ncols
          raise "Matrix must be square"
        end

        a = a.clone
        dim = Math.min(a.nrows, a.ncols)
        ipiv = Pointer(Int32).malloc(dim)

        m = a.nrows
        n = a.ncols
        lda = a.@tda

        LibLapack.{{blas_prefix}}getrf(
          pointerof(m),
          pointerof(n),
          a.@buffer,
          pointerof(lda),
          ipiv,
          out info
        )

        order = dim
        lwork = m * n
        work = Pointer({{dtype}}).malloc(lwork)

        LibLapack.{{blas_prefix}}getri(
          pointerof(order),
          a.@buffer,
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
