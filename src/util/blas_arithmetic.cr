require "../core/vector"
require "../llib/gsl"

module Bottle::Util::VectorMath
  extend self

  def vector_dot(vector : Vector(LibGsl::GslVector), other : Vector(LibGsl::GslVector))
    res = 0.0
    LibGsl.gsl_blas_ddot(vector.ptr, other.ptr, pointerof(res))
    return res
  end

  def vector_norm(vector : Vector(LibGsl::GslVector))
    LibGsl.gsl_blas_dnrm2(vector.ptr)
  end
end
