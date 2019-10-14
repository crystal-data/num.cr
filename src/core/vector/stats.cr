require "../../libs/gsl"
require "../../vector/*"

module Bottle::Core::VectorStats
  extend self

  def vector_max(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_max(a)
  end

  def vector_min(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_min(a)
  end

  def vector_ptpv(a : Pointer(LibGsl::GslVector))
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_vector_minmax(a, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  def vector_ptp(a : Pointer(LibGsl::GslVector))
    mn, mx = vector_ptpv(a)
    return mx - mn
  end

  def vector_idxmax(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_max_index(a)
  end

  def vector_idxmin(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_min_index(a)
  end

  def vector_ptpidx(a : Pointer(LibGsl::GslVector))
    imin : UInt64 = 0
    imax : UInt64 = 0
    LibGsl.gsl_vector_minmax_index(a, pointerof(imin), pointerof(imax))
    return imin, imax
  end
end
