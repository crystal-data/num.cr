require "../../libs/gsl"
require "../../vector/*"

macro statistics_abstract(type_, prefix)
  module Bottle::Core::VectorStats
    extend self

    def vector_max(a : Pointer({{ type_ }}))
      return LibGsl.{{ prefix }}_max(a)
    end

    def vector_min(a : Pointer({{ type_ }}))
      return LibGsl.{{ prefix }}_min(a)
    end

    def vector_ptpv(a : Pointer({{ type_ }}))
      min_out = 0.0
      max_out = 0.0
      LibGsl.{{ prefix }}_minmax(a, pointerof(min_out), pointerof(max_out))
      return min_out, max_out
    end

    def vector_ptp(a : Pointer({{ type_ }}))
      mn, mx = vector_ptpv(a)
      return mx - mn
    end

    def vector_idxmax(a : Pointer({{ type_ }}))
      return LibGsl.{{ prefix }}_max_index(a)
    end

    def vector_idxmin(a : Pointer({{ type_ }}))
      return LibGsl.{{ prefix }}_min_index(a)
    end

    def vector_ptpidx(a : Pointer({{ type_ }}))
      imin : UInt64 = 0
      imax : UInt64 = 0
      LibGsl.{{ prefix }}_minmax_index(a, pointerof(imin), pointerof(imax))
      return imin, imax
    end
  end
end

statistics_abstract LibGsl::GslVector, gsl_vector
statistics_abstract LibGsl::GslVectorInt, gsl_vector_int
