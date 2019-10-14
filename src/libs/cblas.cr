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

  fun ddot = cblas_ddot(n : Integer, x : Double*, incx : Integer, y : Double*, incy : Integer) : Double
  fun dnrm2 = cblas_dnrm2(n : Integer, x : Double*, incx : Integer) : Double
  fun dscal = cblas_dscal(n : Integer, da : Double, dx : Double*, incx : Integer)
  fun dasum = cblas_dasum(n : Integer, dx : Double*, incx : Integer) : Double
  fun idamax = cblas_idamax(n : Integer, dx : Double*, incx : Integer) : Integer
end
