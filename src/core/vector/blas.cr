# require "../core/vector"
# require "../llib/gsl"
# require "../llib/cblas"
#
# module Bottle::Core::VectorBlas
#   extend self
#
#   def vector_dot(vector : Vector(LibGsl::GslVector), other : Vector(LibGsl::GslVector))
#     LibCblas.ddot(vector.size, vector.@obj.data, vector.stride, other.@obj.data, other.stride)
#   end
#
#   def vector_norm(vector : Vector(LibGsl::GslVector))
#     LibCblas.snrm2(vector.size, vector.@obj.data, vector.stride)
#   end
# end
