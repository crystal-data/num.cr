{% if flag?(:openblas) %}
  @[Link("openblas")]
{% elsif flag?(:accelerate) %}
  @[Link(framework: "Accelerate")]
{% elsif flag?(:darwin) %}
  @[Link(framework: "Accelerate")]
{% else %}
  @[Link("cblas")]
{% end %}
lib LibCblas
  struct ComplexDouble
    re : LibC::Double
    im : LibC::Double
  end

  struct ComplexFloat
    re : LibC::Float
    im : LibC::Float
  end

  enum MatrixLayout : LibC::Int
    ROW_MAJOR = 101
    COL_MAJOR = 102
  end
  ROW_MAJOR = MatrixLayout::ROW_MAJOR
  COL_MAJOR = MatrixLayout::COL_MAJOR

  alias Blasint = LibC::Int

  enum CblasTranspose
    CblasNoTrans     = 111
    CblasTrans       = 112
    CblasConjTrans   = 113
    CblasConjNoTrans = 114
  end

  enum CblasUplo
    CblasUpper = 121
    CblasLower = 122
  end

  enum CblasDiag
    CblasNonUnit = 131
    CblasUnit    = 132
  end

  enum CblasSide
    CblasLeft  = 141
    CblasRight = 142
  end

  fun set_num_threads = openblas_set_num_threads(num_threads : LibC::Int)
  fun get_num_threads = openblas_get_num_threads : LibC::Int
  fun get_num_procs = openblas_get_num_procs : LibC::Int
  fun get_config = openblas_get_config : LibC::Char*
  fun get_corename = openblas_get_corename : LibC::Char*
  fun get_parallel = openblas_get_parallel : LibC::Int
  fun sdsdot = cblas_sdsdot(n : Blasint, alpha : LibC::Float, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint) : LibC::Float
  fun dsdot = cblas_dsdot(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint) : LibC::Double
  fun sdot = cblas_sdot(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint) : LibC::Float
  fun ddot = cblas_ddot(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint) : LibC::Double
  fun cdotu = cblas_cdotu(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint) : ComplexFloat
  fun cdotc = cblas_cdotc(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint) : ComplexFloat
  fun zdotu = cblas_zdotu(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint) : ComplexDouble
  fun zdotc = cblas_zdotc(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint) : ComplexDouble
  fun cdotu_sub = cblas_cdotu_sub(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint, ret : ComplexFloat*)
  fun cdotc_sub = cblas_cdotc_sub(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint, ret : ComplexFloat*)
  fun zdotu_sub = cblas_zdotu_sub(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint, ret : ComplexDouble*)
  fun zdotc_sub = cblas_zdotc_sub(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint, ret : ComplexDouble*)
  fun sasum = cblas_sasum(n : Blasint, x : LibC::Float*, incx : Blasint) : LibC::Float
  fun dasum = cblas_dasum(n : Blasint, x : LibC::Double*, incx : Blasint) : LibC::Double
  fun scasum = cblas_scasum(n : Blasint, x : LibC::Float*, incx : Blasint) : LibC::Float
  fun dzasum = cblas_dzasum(n : Blasint, x : LibC::Double*, incx : Blasint) : LibC::Double
  fun snrm2 = cblas_snrm2(n : Blasint, x : LibC::Float*, inc_x : Blasint) : LibC::Float
  fun dnrm2 = cblas_dnrm2(n : Blasint, x : LibC::Double*, inc_x : Blasint) : LibC::Double
  fun scnrm2 = cblas_scnrm2(n : Blasint, x : LibC::Float*, inc_x : Blasint) : LibC::Float
  fun dznrm2 = cblas_dznrm2(n : Blasint, x : LibC::Double*, inc_x : Blasint) : LibC::Double
  fun isamax = cblas_isamax(n : Blasint, x : LibC::Float*, incx : Blasint) : LibC::SizeT
  fun idamax = cblas_idamax(n : Blasint, x : LibC::Double*, incx : Blasint) : LibC::SizeT
  fun icamax = cblas_icamax(n : Blasint, x : LibC::Float*, incx : Blasint) : LibC::SizeT
  fun izamax = cblas_izamax(n : Blasint, x : LibC::Double*, incx : Blasint) : LibC::SizeT
  fun saxpy = cblas_saxpy(n : Blasint, alpha : LibC::Float, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun daxpy = cblas_daxpy(n : Blasint, alpha : LibC::Double, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun caxpy = cblas_caxpy(n : Blasint, alpha : LibC::Float*, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun zaxpy = cblas_zaxpy(n : Blasint, alpha : LibC::Double*, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun scopy = cblas_scopy(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun dcopy = cblas_dcopy(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun ccopy = cblas_ccopy(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun zcopy = cblas_zcopy(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun sswap = cblas_sswap(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun dswap = cblas_dswap(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun cswap = cblas_cswap(n : Blasint, x : LibC::Float*, incx : Blasint, y : LibC::Float*, incy : Blasint)
  fun zswap = cblas_zswap(n : Blasint, x : LibC::Double*, incx : Blasint, y : LibC::Double*, incy : Blasint)
  fun srot = cblas_srot(n : Blasint, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, c : LibC::Float, s : LibC::Float)
  fun drot = cblas_drot(n : Blasint, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, c : LibC::Double, s : LibC::Double)
  fun srotg = cblas_srotg(a : LibC::Float*, b : LibC::Float*, c : LibC::Float*, s : LibC::Float*)
  fun drotg = cblas_drotg(a : LibC::Double*, b : LibC::Double*, c : LibC::Double*, s : LibC::Double*)
  fun srotm = cblas_srotm(n : Blasint, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, p : LibC::Float*)
  fun drotm = cblas_drotm(n : Blasint, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, p : LibC::Double*)
  fun srotmg = cblas_srotmg(d1 : LibC::Float*, d2 : LibC::Float*, b1 : LibC::Float*, b2 : LibC::Float, p : LibC::Float*)
  fun drotmg = cblas_drotmg(d1 : LibC::Double*, d2 : LibC::Double*, b1 : LibC::Double*, b2 : LibC::Double, p : LibC::Double*)
  fun sscal = cblas_sscal(n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint)
  fun dscal = cblas_dscal(n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint)
  fun cscal = cblas_cscal(n : Blasint, alpha : LibC::Float*, x : LibC::Float*, inc_x : Blasint)
  fun zscal = cblas_zscal(n : Blasint, alpha : LibC::Double*, x : LibC::Double*, inc_x : Blasint)
  fun csscal = cblas_csscal(n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint)
  fun zdscal = cblas_zdscal(n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint)
  fun sgemv = cblas_sgemv(order : MatrixLayout, trans : CblasTranspose, m : Blasint, n : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, x : LibC::Float*, incx : Blasint, beta : LibC::Float, y : LibC::Float*, incy : Blasint)
  fun dgemv = cblas_dgemv(order : MatrixLayout, trans : CblasTranspose, m : Blasint, n : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, x : LibC::Double*, incx : Blasint, beta : LibC::Double, y : LibC::Double*, incy : Blasint)
  fun cgemv = cblas_cgemv(order : MatrixLayout, trans : CblasTranspose, m : Blasint, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, x : LibC::Float*, incx : Blasint, beta : LibC::Float*, y : LibC::Float*, incy : Blasint)
  fun zgemv = cblas_zgemv(order : MatrixLayout, trans : CblasTranspose, m : Blasint, n : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, x : LibC::Double*, incx : Blasint, beta : LibC::Double*, y : LibC::Double*, incy : Blasint)
  fun sger = cblas_sger(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*, lda : Blasint)
  fun dger = cblas_dger(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*, lda : Blasint)
  fun cgeru = cblas_cgeru(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Float*, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*, lda : Blasint)
  fun cgerc = cblas_cgerc(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Float*, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*, lda : Blasint)
  fun zgeru = cblas_zgeru(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Double*, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*, lda : Blasint)
  fun zgerc = cblas_zgerc(order : MatrixLayout, m : Blasint, n : Blasint, alpha : LibC::Double*, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*, lda : Blasint)
  fun strsv = cblas_strsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun dtrsv = cblas_dtrsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun ctrsv = cblas_ctrsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun ztrsv = cblas_ztrsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun strmv = cblas_strmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun dtrmv = cblas_dtrmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun ctrmv = cblas_ctrmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun ztrmv = cblas_ztrmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun ssyr = cblas_ssyr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, a : LibC::Float*, lda : Blasint)
  fun dsyr = cblas_dsyr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, a : LibC::Double*, lda : Blasint)
  fun cher = cblas_cher(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, a : LibC::Float*, lda : Blasint)
  fun zher = cblas_zher(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, a : LibC::Double*, lda : Blasint)
  fun ssyr2 = cblas_ssyr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*, lda : Blasint)
  fun dsyr2 = cblas_dsyr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*, lda : Blasint)
  fun cher2 = cblas_cher2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float*, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*, lda : Blasint)
  fun zher2 = cblas_zher2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double*, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*, lda : Blasint)
  fun sgbmv = cblas_sgbmv(order : MatrixLayout, trans_a : CblasTranspose, m : Blasint, n : Blasint, kl : Blasint, ku : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float, y : LibC::Float*, inc_y : Blasint)
  fun dgbmv = cblas_dgbmv(order : MatrixLayout, trans_a : CblasTranspose, m : Blasint, n : Blasint, kl : Blasint, ku : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double, y : LibC::Double*, inc_y : Blasint)
  fun cgbmv = cblas_cgbmv(order : MatrixLayout, trans_a : CblasTranspose, m : Blasint, n : Blasint, kl : Blasint, ku : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float*, y : LibC::Float*, inc_y : Blasint)
  fun zgbmv = cblas_zgbmv(order : MatrixLayout, trans_a : CblasTranspose, m : Blasint, n : Blasint, kl : Blasint, ku : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double*, y : LibC::Double*, inc_y : Blasint)
  fun ssbmv = cblas_ssbmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, k : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float, y : LibC::Float*, inc_y : Blasint)
  fun dsbmv = cblas_dsbmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, k : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double, y : LibC::Double*, inc_y : Blasint)
  fun stbmv = cblas_stbmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun dtbmv = cblas_dtbmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun ctbmv = cblas_ctbmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun ztbmv = cblas_ztbmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun stbsv = cblas_stbsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun dtbsv = cblas_dtbsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun ctbsv = cblas_ctbsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint)
  fun ztbsv = cblas_ztbsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, k : Blasint, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint)
  fun stpmv = cblas_stpmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint)
  fun dtpmv = cblas_dtpmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint)
  fun ctpmv = cblas_ctpmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint)
  fun ztpmv = cblas_ztpmv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint)
  fun stpsv = cblas_stpsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint)
  fun dtpsv = cblas_dtpsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint)
  fun ctpsv = cblas_ctpsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint)
  fun ztpsv = cblas_ztpsv(order : MatrixLayout, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, n : Blasint, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint)
  fun ssymv = cblas_ssymv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float, y : LibC::Float*, inc_y : Blasint)
  fun dsymv = cblas_dsymv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double, y : LibC::Double*, inc_y : Blasint)
  fun chemv = cblas_chemv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float*, y : LibC::Float*, inc_y : Blasint)
  fun zhemv = cblas_zhemv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double*, y : LibC::Double*, inc_y : Blasint)
  fun sspmv = cblas_sspmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float, y : LibC::Float*, inc_y : Blasint)
  fun dspmv = cblas_dspmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double, y : LibC::Double*, inc_y : Blasint)
  fun sspr = cblas_sspr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, ap : LibC::Float*)
  fun dspr = cblas_dspr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, ap : LibC::Double*)
  fun chpr = cblas_chpr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, a : LibC::Float*)
  fun zhpr = cblas_zhpr(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, a : LibC::Double*)
  fun sspr2 = cblas_sspr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, a : LibC::Float*)
  fun dspr2 = cblas_dspr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, a : LibC::Double*)
  fun chpr2 = cblas_chpr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float*, x : LibC::Float*, inc_x : Blasint, y : LibC::Float*, inc_y : Blasint, ap : LibC::Float*)
  fun zhpr2 = cblas_zhpr2(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double*, x : LibC::Double*, inc_x : Blasint, y : LibC::Double*, inc_y : Blasint, ap : LibC::Double*)
  fun chbmv = cblas_chbmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float*, y : LibC::Float*, inc_y : Blasint)
  fun zhbmv = cblas_zhbmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, k : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double*, y : LibC::Double*, inc_y : Blasint)
  fun chpmv = cblas_chpmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Float*, ap : LibC::Float*, x : LibC::Float*, inc_x : Blasint, beta : LibC::Float*, y : LibC::Float*, inc_y : Blasint)
  fun zhpmv = cblas_zhpmv(order : MatrixLayout, uplo : CblasUplo, n : Blasint, alpha : LibC::Double*, ap : LibC::Double*, x : LibC::Double*, inc_x : Blasint, beta : LibC::Double*, y : LibC::Double*, inc_y : Blasint)
  fun sgemm = cblas_sgemm(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun dgemm = cblas_dgemm(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun cgemm = cblas_cgemm(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun cgemm3m = cblas_cgemm3m(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun zgemm = cblas_zgemm(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : ComplexDouble*, a : ComplexDouble*, lda : Blasint, b : ComplexDouble*, ldb : Blasint, beta : ComplexDouble*, c : ComplexDouble*, ldc : Blasint)
  fun zgemm3m = cblas_zgemm3m(order : MatrixLayout, trans_a : CblasTranspose, trans_b : CblasTranspose, m : Blasint, n : Blasint, k : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double*, c : LibC::Double*, ldc : Blasint)
  fun ssymm = cblas_ssymm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun dsymm = cblas_dsymm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun csymm = cblas_csymm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun zsymm = cblas_zsymm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : ComplexDouble*, a : ComplexDouble*, lda : Blasint, b : ComplexDouble*, ldb : Blasint, beta : ComplexDouble*, c : ComplexDouble*, ldc : Blasint)
  fun ssyrk = cblas_ssyrk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun dsyrk = cblas_dsyrk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun csyrk = cblas_csyrk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun zsyrk = cblas_zsyrk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, beta : LibC::Double*, c : LibC::Double*, ldc : Blasint)
  fun ssyr2k = cblas_ssyr2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun dsyr2k = cblas_dsyr2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun csyr2k = cblas_csyr2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun zsyr2k = cblas_zsyr2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double*, c : LibC::Double*, ldc : Blasint)
  fun strmm = cblas_strmm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint)
  fun dtrmm = cblas_dtrmm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint)
  fun ctrmm = cblas_ctrmm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint)
  fun ztrmm = cblas_ztrmm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : ComplexDouble*, a : ComplexDouble*, lda : Blasint, b : ComplexDouble*, ldb : Blasint)
  fun strsm = cblas_strsm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint)
  fun dtrsm = cblas_dtrsm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint)
  fun ctrsm = cblas_ctrsm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint)
  fun ztrsm = cblas_ztrsm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, trans_a : CblasTranspose, diag : CblasDiag, m : Blasint, n : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint)
  fun chemm = cblas_chemm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float*, c : LibC::Float*, ldc : Blasint)
  fun zhemm = cblas_zhemm(order : MatrixLayout, side : CblasSide, uplo : CblasUplo, m : Blasint, n : Blasint, alpha : ComplexDouble*, a : ComplexDouble*, lda : Blasint, b : ComplexDouble*, ldb : Blasint, beta : ComplexDouble*, c : ComplexDouble*, ldc : Blasint)
  fun cherk = cblas_cherk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float, a : LibC::Float*, lda : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun zherk = cblas_zherk(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double, a : LibC::Double*, lda : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun cher2k = cblas_cher2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Float*, a : LibC::Float*, lda : Blasint, b : LibC::Float*, ldb : Blasint, beta : LibC::Float, c : LibC::Float*, ldc : Blasint)
  fun zher2k = cblas_zher2k(order : MatrixLayout, uplo : CblasUplo, trans : CblasTranspose, n : Blasint, k : Blasint, alpha : LibC::Double*, a : LibC::Double*, lda : Blasint, b : LibC::Double*, ldb : Blasint, beta : LibC::Double, c : LibC::Double*, ldc : Blasint)
  fun xerbla = cblas_xerbla(p : Blasint, rout : LibC::Char*, form : LibC::Char*, ...)
  fun saxpby = cblas_saxpby(n : Blasint, alpha : LibC::Float, x : LibC::Float*, incx : Blasint, beta : LibC::Float, y : LibC::Float*, incy : Blasint)
  fun daxpby = cblas_daxpby(n : Blasint, alpha : LibC::Double, x : LibC::Double*, incx : Blasint, beta : LibC::Double, y : LibC::Double*, incy : Blasint)
  fun caxpby = cblas_caxpby(n : Blasint, alpha : LibC::Float*, x : LibC::Float*, incx : Blasint, beta : LibC::Float*, y : LibC::Float*, incy : Blasint)
  fun zaxpby = cblas_zaxpby(n : Blasint, alpha : LibC::Double*, x : LibC::Double*, incx : Blasint, beta : LibC::Double*, y : LibC::Double*, incy : Blasint)
  fun somatcopy = cblas_somatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Float, a : LibC::Float*, clda : Blasint, b : LibC::Float*, cldb : Blasint)
  fun domatcopy = cblas_domatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Double, a : LibC::Double*, clda : Blasint, b : LibC::Double*, cldb : Blasint)
  fun comatcopy = cblas_comatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Float*, a : LibC::Float*, clda : Blasint, b : LibC::Float*, cldb : Blasint)
  fun zomatcopy = cblas_zomatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Double*, a : LibC::Double*, clda : Blasint, b : LibC::Double*, cldb : Blasint)
  fun simatcopy = cblas_simatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Float, a : LibC::Float*, clda : Blasint, cldb : Blasint)
  fun dimatcopy = cblas_dimatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Double, a : LibC::Double*, clda : Blasint, cldb : Blasint)
  fun cimatcopy = cblas_cimatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Float*, a : LibC::Float*, clda : Blasint, cldb : Blasint)
  fun zimatcopy = cblas_zimatcopy(corder : MatrixLayout, ctrans : CblasTranspose, crows : Blasint, ccols : Blasint, calpha : LibC::Double*, a : LibC::Double*, clda : Blasint, cldb : Blasint)
  fun sgeadd = cblas_sgeadd(corder : MatrixLayout, crows : Blasint, ccols : Blasint, calpha : LibC::Float, a : LibC::Float*, clda : Blasint, cbeta : LibC::Float, c : LibC::Float*, cldc : Blasint)
  fun dgeadd = cblas_dgeadd(corder : MatrixLayout, crows : Blasint, ccols : Blasint, calpha : LibC::Double, a : LibC::Double*, clda : Blasint, cbeta : LibC::Double, c : LibC::Double*, cldc : Blasint)
  fun cgeadd = cblas_cgeadd(corder : MatrixLayout, crows : Blasint, ccols : Blasint, calpha : LibC::Float*, a : LibC::Float*, clda : Blasint, cbeta : LibC::Float*, c : LibC::Float*, cldc : Blasint)
  fun zgeadd = cblas_zgeadd(corder : MatrixLayout, crows : Blasint, ccols : Blasint, calpha : LibC::Double*, a : LibC::Double*, clda : Blasint, cbeta : LibC::Double*, c : LibC::Double*, cldc : Blasint)
end
