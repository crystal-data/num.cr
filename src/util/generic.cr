# require "../core/vector"
# require "../llib/gsl"
#
# DOUBLE_TYPE = LibGsl::GslVector
# INTEGER_TYPE = LibGsl::GslVectorInt
#
# macro abstract_cast(type_, prefix)
#   module Bottle::Util::Cast
#     extend self
#
#     def vector_to_double(vector : Vector({{ type_ }}))
#       if vector.is_a?(DOUBLE_TYPE)
#         return vector.as(Vector(LibGsl::GslVector))
#       end
#       ptv = LibGsl.gsl_vector_alloc(vector.size)
#       (0...vector.size).each do |i|
#         LibGsl.gsl_vector_set(ptv, i, LibGsl.{{ prefix }}_get(vector.ptr, i))
#       end
#       vector = Vector.new(ptv)
#       return vector.as(Vector(LibGsl::GslVector))
#     end
#
#     def vector_to_int(vector : Vector({{ type_ }}))
#       if vector.is_a?(INTEGER_TYPE)
#         return vector
#       end
#       ptv = LibGsl.gsl_vector_int_alloc(vector.size)
#       (0...vector.size).each do |i|
#         LibGsl.gsl_vector_int_set(ptv, i, LibGsl.{{ prefix }}_get(vector.ptr, i))
#       end
#       return Vector.new ptv
#     end
#   end
# end
#
# abstract_cast LibGsl::GslVectorInt, gsl_vector_int
# abstract_cast LibGsl::GslVector, gsl_vector
