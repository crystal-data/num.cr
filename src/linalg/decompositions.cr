require "../tensor/tensor"
require "../libs/lapack"
require "../libs/cblas"

module Bottle::LinAlg
  private def assert_square_matrix(a)
    if a.ndims != 2 || a.shape[0] != a.shape[1]
      raise Internal::Exceptions::ShapeError.new("Input must be a square matrix")
    end
  end

  private def assert_matrix(a)
    if a.ndims != 2
      raise Internal::Exceptions::ShapeError.new("Input must be a matrix")
    end
  end

  macro linalg(dtype, prefix)

    def cholesky(a : Tensor({{dtype}}))
      assert_square_matrix(a)
      u8uplo = 'L'.ord.to_u8
      a = a.dup('F')
      n = a.shape[0]
      lda = a.strides[1]

      LibLapack.{{prefix}}potrf(
        pointerof(u8uplo),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        out info
      )

      if info < 0
        raise "Illegal argument provided"
      elsif info > 1
        raise "Tensor is not positive definite"
      else
        return Creation.tril(a)
      end
    end

    private def query_work_size(m, n, aptr : Pointer({{dtype}}), lda, tauptr, jpvtptr)
      findwork = -1
      LibLapack.{{prefix}}geqrf(
        pointerof(m),
        pointerof(n),
        aptr,
        pointerof(lda),
        tauptr,
        jpvtptr,
        pointerof(findwork),
        out info
      )
      Int32.new(jpvtptr.value)
    end

    def qr(a : Tensor({{dtype}}))
      assert_matrix(a)
      m, n = a.shape
      a = a.dup('F')
      lda = a.strides[1]
      k = {m, n}.min
      tau = Tensor({{dtype}}).new([k])
      jpvt = Pointer({{dtype}}).malloc(1)

      lwork = query_work_size(m, n, a.buffer, lda, tau.buffer, jpvt)
      work = Pointer({{dtype}}).malloc(lwork)

      LibLapack.{{prefix}}geqrf(
        pointerof(m),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        tau.buffer,
        work,
        pointerof(lwork),
        out info,
      )

      r = Creation.triu(a.dup('F'))

      LibLapack.{{prefix}}orgqr(
        pointerof(m),
        pointerof(n),
        pointerof(k),
        a.buffer,
        pointerof(lda),
        tau.buffer,
        work,
        pointerof(n),
        out info2
      )

      {a, r}
    end

    def svd(a : Tensor({{dtype}}))
      assert_matrix(a)
      jobu = 'A'.ord.to_u8
      jobvt = 'A'.ord.to_u8
      m, n = a.shape
      a = a.dup('F')

      lda = a.strides[1]
      k = {m, n}.min
      s = Tensor({{dtype}}).new([k])
      u = Tensor({{dtype}}).new([m, m])
      ldu = m
      vt = Tensor({{dtype}}).new([n, n])
      ldvt = n
      lwork = 3 * k + {m, n}.max + 5 * k
      work = Pointer({{dtype}}).malloc(lwork)

      LibLapack.{{prefix}}gesvd(
        pointerof(jobu),
        pointerof(jobvt),
        pointerof(m),
        pointerof(n),
        a.buffer,
        pointerof(lda),
        s.buffer,
        u.buffer,
        pointerof(ldu),
        vt.buffer,
        pointerof(ldvt),
        work,
        pointerof(lwork),
        out info
      )
      {u, s, vt}
    end
  end

  linalg(Float64, d)
  linalg(Float32, s)
end
