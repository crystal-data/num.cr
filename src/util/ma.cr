# require "../llib/gsl"
# require "../core/vector"
#
# macro mask_abstract(type_, prefix)
#   module Bottle::Util::VectorMask
#     include Bottle::Util::Exceptions
#     extend self
#
#     def vector_nonzero(a : Pointer({{ type_ }}))
#       ptv = LibMa.{{ prefix }}_ma_nonzero(a)
#       return Vector.new ptv
#     end
#
#     def vector_gt(a : Vector({{ type_ }}), b : Vector({{ type_ }}))
#       ptv = LibMa.{{ prefix }}_ma_gt(a.ptr, b.ptr)
#       return Vector.new ptv
#     end
#
#   end
# end
#
# mask_abstract LibGsl::GslVector, gsl_vector
# mask_abstract LibGsl::GslVectorInt, gsl_vector_int
