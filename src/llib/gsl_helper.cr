require "./lib_gsl"
require "../core/vector"

module Bottle::GslVec
  extend self
  def newv(data : Indexable(Float64))
    sz = data.size
    v = LibGsl.gsl_vector_alloc(sz)
    data.each_with_index { |e, i| LibGsl.gsl_vector_set(v, i, e) }
    return v, sz
  end

  def max(vector)
    return LibGsl.gsl_blas_idamax(vector)
  end

  def add(vector, vector)

  end

  def get(vector, ii : Int32)
    return LibGsl.gsl_vector_get(vector, ii)
  end

  def get(vector, ii : Range(Int32, Int32))
    start = ii.begin
    close = ii.end
    view = LibGsl.gsl_vector_subvector(vector, start, close - start)
    return VectorView(Float64).new view, close - start
  end

  def get(vector, ii : Iterable(Int32))
    return Vector.new ii.map { |e| LibGsl.gsl_vector_get(vector, e) }
  end

  def set(vector, ii : Int32, value : Float64)
    LibGsl.gsl_vector_set(vector, ii, value)
  end

  def set(vector, ii : Iterable(Int32), vv : Iterable(Float64))
    ii.zip(vv).each { |i, j| LibGsl.gsl_vector_set(vector, i, j) }
  end

  def slice_all(vector, size, io)
    io << "Vector[" << vector.value.data.to_slice(size).join(", ") << "]"
  end
end
