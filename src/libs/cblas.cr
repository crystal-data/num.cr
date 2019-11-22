require "./dtype"

{% if flag?(:darwin) %}
  @[Link(framework: "Accelerate")]
{% else %}
  @[Link("openblas")]
{% end %}
lib LibCblas
  alias Integer = LibC::Int
  alias Real = LibC::Float
  alias Double = LibC::Double
  alias Logical = LibC::Char
  alias Ftnlen = LibC::Int
  alias LFp = Pointer(Void)
  alias UInt = LibC::SizeT
  alias Indexer = UInt64 | Int32
  alias BNum = Int32 | Float64 | Float32

  enum MatrixLayout : Integer
    RowMajor = 101
    ColMajor = 102
  end

  enum MatrixTranspose
    NoTrans     = 111
    Trans       = 112
    ConjTrans   = 113
    ConjNoTrans = 114
  end

  struct ComplexDouble
    re : LibC::Double
    im : LibC::Double
  end

  struct ComplexFloat
    re : LibC::Float
    im : LibC::Float
  end

  # Level 1
  fun srotg = cblas_srotg(da : Real*, db : Real*, c : Real*, s : Real*)
  fun drotg = cblas_drotg(da : Double*, db : Double*, c : Double*, s : Double*)

  fun saxpy = cblas_saxpy(n : Integer, sa : Real, sx : Real*, incx : Integer, sy : Real*, incy : Integer)
  fun daxpy = cblas_daxpy(n : Integer, sa : Double, sx : Double*, incx : Integer, sy : Double*, incy : Integer)

  fun drot = cblas_drot(n : Integer, dx : Double*, incx : Integer, dy : Double*, incy : Integer, c : Double*, s : Double*)
  fun srot = cblas_srot(n : Integer, dx : Real*, incx : Integer, dy : Real*, incy : Integer, c : Real*, s : Real*)

  fun ddot = cblas_ddot(n : Integer, x : Double*, incx : Integer, y : Double*, incy : Integer) : Double
  fun sdot = cblas_sdot(n : Integer, x : Real*, incx : Integer, y : Real*, incy : Integer) : Real
  fun zdot = cblas_zdotu(n : Integer, x : Double*, incx : Integer, y : Double*, incy : Integer) : ComplexDouble

  fun dnrm2 = cblas_dnrm2(n : Integer, x : Double*, incx : Integer) : Double
  fun snrm2 = cblas_snrm2(n : Integer, x : Real*, incx : Integer) : Real

  fun dscal = cblas_dscal(n : Integer, da : Double, dx : Double*, incx : Integer)
  fun sscal = cblas_sscal(n : Integer, da : Real*, dx : Real*, incx : Integer)

  fun dasum = cblas_dasum(n : Integer, dx : Double*, incx : Integer) : Double
  fun sasum = cblas_sasum(n : Integer, dx : Real*, incx : Integer) : Real

  fun idamax = cblas_idamax(n : Integer, dx : Double*, incx : Integer) : Integer
  fun isamax = cblas_isamax(n : Integer, dx : Real*, incx : Integer) : Integer

  # Level 2
  fun dgemv = cblas_dgemv(order : MatrixLayout, trans : MatrixTranspose, m : Integer,
                          n : Integer, alpha : Double, a : Double*, lda : Integer, x : Double*, incx : Integer, beta : Double, y : Double*, incy : Integer)
  fun dger = cblas_dger(order : MatrixLayout, m : Integer, n : Integer, alpha : Double, x : Double*,
                        incx : Integer, y : Double*, incy : Integer, a : Double*, lda : Integer)

  fun sger = cblas_sger(order : MatrixLayout, m : Integer, n : Integer, alpha : Real, x : Real*, incx : Integer, y : Real*, incy : Integer, a : Real*, lda : Integer)

  # Level 3
  fun dgemm = cblas_dgemm(order : MatrixLayout, trans_a : MatrixTranspose, trans_b : MatrixTranspose,
                          m : Integer, n : Integer, k : Integer, alpha : Double, a : Double*, lda : Integer, b : Double*,
                          ldb : Integer, beta : Double, c : Double*, ldc : Integer)

  fun sgemm = cblas_sgemm(order : MatrixLayout, trans_a : MatrixTranspose, trans_b : MatrixTranspose,
                          m : Integer, n : Integer, k : Integer, alpha : Real, a : Real*, lda : Integer, b : Real*,
                          ldb : Integer, beta : Real, c : Real*, ldc : Integer)

  fun zgemm = cblas_zgemm(order : MatrixLayout, trans_a : MatrixTranspose, trans_b : MatrixTranspose,
                          m : Integer, n : Integer, k : Integer, alpha : Double, a : Double*, lda : Integer, b : Double*,
                          ldb : Integer, beta : Double, c : Double*, ldc : Integer)
end
