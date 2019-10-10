require "../llib/lib_gsl"
require "../core/vector"

module Bottle::Util::Indexing
  extend self

  def _take_vec_at_index(a, index : Int32)
    return LibGsl.gsl_vector_get(a, index)
  end

  def _set_vec_at_index(a, index : Int32, value : Number)
    LibGsl.gsl_vector_set(a, index, value)
  end

  def _take_vec_at_indexes(a, indexes : Iterable(Int32))
    ptv = LibGsl.gsl_vector_alloc(indexes.size)
    indexes.each_with_index do |el, index|
      val = LibGsl.gsl_vector_get(a, el)
      _set_vec_at_index(ptv, index, val)
    end
    return Vector.new ptv.value, indexes.size
  end

  def _take_vec_at_range(a, range)
    ii = range.begin
    jj_ = range.end
    jj = range.excludes_end? ? jj_ - 1 : jj_
    view = LibGsl.gsl_vector_subvector(a, ii, jj - ii)
    return Vector.new(view.vector, jj - ii)
  end

end
