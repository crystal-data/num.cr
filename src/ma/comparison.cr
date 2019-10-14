require "../llib/gsl"
require "../util/exceptions"
require "../core/vector"
require "../llib/bottle"

macro mask_comparison_abstract(type_, dtype, prefix)
  module Bottle::Ma::Comparison
    include Bottle::Util::Exceptions
    extend self

    def vector_equal(a : Vector({{ type_ }}), b : Vector)
      ptv = LibBottle.{{ type_ }}_ma_equal(a.ptr, b.ptr)
      return Vector.new ptv, ptv.value.data
    end
  end
end

mask_comparison_abstract LibGsl::GslVector, Float64, gsl_vector
