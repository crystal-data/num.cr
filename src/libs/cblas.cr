{% if flag?(:darwin) %}
  @[Link(framework: "Accelerate")]
{% else %}
  @[Link("cblas")]
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
  fun snrm2 = cblas_snrm2(n : Integer, x : Double*, incx : Integer) : Double
end
