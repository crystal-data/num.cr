require "../llib/lib_gsl"
require "../core/vector"

module Bottle::Util::Arithmetic
  extend self

  def _vec_max(a)
    return LibGsl.gsl_vector_max(a)
  end

  def _vec_min(a)
    return LibGsl.gsl_vector_min(a)
  end

  def _vec_ptpv(a)
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_vector_minmax(a, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  def _vec_ptp(a)
    mn, mx = ptpv(a)
    return mx - mn
  end

  def _vec_idxmax(a)
    return LibGsl.gsl_vector_max_index(a)
  end

  def _vec_idxmin(a)
    return LibGsl.gsl_vector_min_index(a)
  end

  def _vec_ptpidx(a)
    imin : UInt64 = 0
    imax : UInt64 = 0
    LibGsl.gsl_vector_minmax_index(a, pointerof(imin), pointerof(imax))
    return imin, imax
  end
end
