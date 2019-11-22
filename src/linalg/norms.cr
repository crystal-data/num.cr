require "../tensor/tensor"
require "../libs/lapack"
require "../libs/cblas"

module Bottle::LinAlg
  macro linalg(dtype, prefix)
    def norm(a : Tensor({{dtype}}), order='F')
      assert_matrix(a)
      a = a.dup('F')
      nrm = order.ord.to_u8
      m, n = a.shape
      lda = a.strides[1]
      work = Pointer({{dtype}}).malloc(m)
      LibLapack.{{prefix}}lange(
        pointerof(nrm),
        pointerof(m),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        work,
      )
    end

    def cond(a : Tensor({{dtype}}), order='1')
      assert_matrix(a)
      a = a.dup('F')
      m, n = a.shape
      anorm = norm(a, order)

      lda = a.strides[1]
      ipiv = Pointer(Int32).malloc({m, n}.min)

      LibLapack.{{prefix}}getrf(
        pointerof(m),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        ipiv,
        out info
      )

      nrm = order.ord.to_u8
      work = Pointer({{dtype}}).malloc(4 * n)
      iwork = Pointer(Int32).malloc(n)

      LibLapack.{{prefix}}gecon(
        pointerof(nrm),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        pointerof(anorm),
        out rcond,
        work,
        iwork,
        out info2
      )
      1 / rcond
    end

    def det(a : Tensor({{dtype}}))
      assert_square_matrix(a)
      a = a.dup('F')
      m, n = a.shape
      lda = a.strides[1]
      ipiv = Pointer(Int32).malloc({m, n}.min)

      LibLapack.{{prefix}}getrf(
        pointerof(m),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        ipiv,
        out info
      )
      ldet = Statistics.prod(a.diag_view)
      detp = 1

      n.times do |j|
        if j+1 != ipiv[j]
          detp = -detp
        end
      end
      ldet * detp
    end
  end

  linalg(Float64, d)
  linalg(Float32, s)
end
