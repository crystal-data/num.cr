require "../llib/gsl"

def vector_type(el : Array(Float64)) : Pointer(LibGsl::GslVector)
  vector = LibGsl.gsl_vector_alloc(el.size)
  el.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
  return vector
end

def vector_type(el : Array(Int32)) : Pointer(LibGsl::GslVectorInt)
  vector = LibGsl.gsl_vector_int_alloc(el.size)
  el.each_with_index { |e, i| LibGsl.gsl_vector_int_set(vector, i, e) }
  return vector
end
