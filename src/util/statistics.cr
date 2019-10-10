require "../llib/lib_gsl"
require "../core/vector"

module Bottle::Util::Statistics
  extend self

  # Vector statistics
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
    mn, mx = _vec_ptpv(a)
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

  # Matrix statistics
  def _mat_max(m)
    return LibGsl.gsl_matrix_max(m)
  end

  def _mat_min(m)
    return LibGsl.gsl_matrix_min(m)
  end

  def _mat_ptpv(m)
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_matrix_minmax(m, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  def _mat_ptp(m)
    mn, mx = _mat_ptpv(m)
    return mx - mn
  end

  def _mat_idxmax(m)
    imax : UInt64 = 0
    jmax : UInt64 = 0
    LibGsl.gsl_matrix_max_index(m, pointerof(imax), pointerof(jmax))
    return imax, jmax
  end

  def _mat_idxmin(m)
    imin : UInt64 = 0
    jmin : UInt64 = 0
    LibGsl.gsl_matrix_min_index(m, pointerof(imin), pointerof(jmin))
    return imin, jmin
  end

  def _mat_ptpidx(m)
    imax : UInt64 = 0
    jmax : UInt64 = 0
    imin : UInt64 = 0
    jmin : UInt64 = 0
    LibGsl.gsl_matrix_minmax_index(m, pointerof(imin), pointerof(jmin), pointerof(imax), pointerof(jmax))
    return imin, jmin, imax, jmax
  end
end
