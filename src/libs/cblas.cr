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

  # Level 1
  fun drotg = cblas_drotg(da : Double, db : Double, c : Double*, s : Double*)
  fun drot = cblas_drot(n : Integer, dx : Double*, incx : Integer, dy : Double*, incy : Integer, c : Double, s : Double)
  fun ddot = cblas_ddot(n : Integer, x : Double*, incx : Integer, y : Double*, incy : Integer) : Double
  fun dnrm2 = cblas_dnrm2(n : Integer, x : Double*, incx : Integer) : Double
  fun dscal = cblas_dscal(n : Integer, da : Double, dx : Double*, incx : Integer)
  fun dasum = cblas_dasum(n : Integer, dx : Double*, incx : Integer) : Double
  fun idamax = cblas_idamax(n : Integer, dx : Double*, incx : Integer) : Integer

  # Level 2
  fun dgemv = cblas_dgemv(order : MatrixLayout, trans : MatrixTranspose, m : Integer,
    n : Integer, alpha : Double, a : Double*, lda : Integer, x : Double*, incx : Integer, beta : Double, y : Double*, incy : Integer)
  fun dger = cblas_dger(order : MatrixLayout, m : Integer, n : Integer, alpha : Double, x : Double*,
    incx : Integer, y : Double*, incy : Integer, a : Double*, lda : Integer)

  # Level 3
  fun dgemm = cblas_dgemm(order : MatrixLayout, trans_a : MatrixTranspose, trans_b : MatrixTranspose,
    m : Integer, n : Integer, k : Integer, alpha : Double, a : Double*, lda : Integer, b : Double*,
    ldb : Integer, beta : Double, c : Double*, ldc : Integer)

end
