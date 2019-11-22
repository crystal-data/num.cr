require "../tensor/tensor"
require "../libs/lapack"
require "../libs/cblas"

module Bottle::LinAlg
  macro linalg(dtype, prefix)

    private def eigcalc(jobz, uplo, n, aptr : Pointer({{dtype}}), lda, wptr, work, lwork)
      LibLapack.{{prefix}}syev(
        pointerof(jobz),
        pointerof(uplo),
        pointerof(n),
        aptr,
        pointerof(lda),
        wptr,
        work,
        pointerof(lwork),
        out info
      )
      info
    end

    def eigh(a : Tensor({{dtype}}))
      assert_square_matrix(a)

      jobz = 'V'.ord.to_u8
      a = a.dup('F')
      uplo = 'L'.ord.to_u8
      n = a.shape[0]
      lda = n
      w = Tensor({{dtype}}).new([n])
      lwork = 3 * n - 1
      work = Pointer({{dtype}}).malloc(lwork)

      eigcalc(jobz, uplo, n, a.buffer, lda, w.buffer, work, lwork)
      {w, a}
    end

    def eigvalsh(a : Tensor({{dtype}}))
      assert_square_matrix(a)

      jobz = 'N'.ord.to_u8
      a = a.dup('F')
      uplo = 'L'.ord.to_u8
      n = a.shape[0]
      lda = n
      w = Tensor({{dtype}}).new([n])
      lwork = 3 * n - 1
      work = Pointer({{dtype}}).malloc(lwork)

      eigcalc(jobz, uplo, n, a.buffer, lda, w.buffer, work, lwork)
      w
    end

    private def eigcalc_nonsymmetric(jobvl, jobvr, n, aptr : Pointer({{dtype}}), lda, wrptr, wlptr, vlptr, ldvl, vrptr, ldvr, work, lwork)
      LibLapack.{{prefix}}geev(
        pointerof(jobvl),
        pointerof(jobvr),
        pointerof(n),
        aptr,
        pointerof(lda),
        wrptr,
        wlptr,
        vlptr,
        pointerof(ldvl),
        vrptr,
        pointerof(ldvr),
        work,
        pointerof(lwork),
        out info
      )
      info
    end

    # def eig(a : Tensor({{dtype}}))
    #   assert_square_matrix(a)
    #   a = a.dup('F')
    #   jobvl = 'V'.ord.to_u8
    #   jobvr = jobvl
    #   n = a.shape[0]
    #   lda = n
    #   wr = Tensor({{dtype}}).new([n])
    #   wl = Tensor({{dtype}}).new([n])
    #   vl = Tensor({{dtype}}).new([n, n])
    #   ldvl = n
    #   vr = Tensor({{dtype}}).new([n])
    #   ldvr = n
    #   lwork = 4 * n
    #   work = Pointer({{dtype}}).malloc(lwork)
    #
    #   info = eigcalc_nonsymmetric(jobvl, jobvr, n, a.buffer, lda, wr.buffer,
    #     wl.buffer, vl.buffer, ldvl, vr.buffer, ldvr, work, lwork)
    #
    #   a
    # end

    def eigvals(a : Tensor({{dtype}}))
      assert_square_matrix(a)
      a = a.dup('F')
      jobvl = 'N'.ord.to_u8
      jobvr = jobvl
      n = a.shape[0]
      lda = n
      wr = Tensor({{dtype}}).new([n])
      wl = Tensor({{dtype}}).new([n])
      vl = Tensor({{dtype}}).new([n, n])
      ldvl = n
      vr = Tensor({{dtype}}).new([n])
      ldvr = n
      lwork = 3 * n
      work = Pointer({{dtype}}).malloc(lwork)

      eigcalc_nonsymmetric(jobvl, jobvr, n, a.buffer, lda, wr.buffer,
        wl.buffer, vl.buffer, ldvl, vr.buffer, ldvr, work, lwork)

      wr
    end

  end

  linalg(Float64, d)
  linalg(Float32, s)
end
