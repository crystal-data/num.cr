require "../llib/lib_gsl"
require "../core/vector"
require "../core/matrix"

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

  def _take_mat_at_index(m, i, j)
    return LibGsl.gsl_matrix_get(m, i, j)
  end

  def _get_vec_at_row(m, i, j)
    vv = LibGsl.gsl_matrix_row(m, i)
    return Vector.new(vv.vector, j)
  end

  # BUG this needs to use `gsl_matrix_column`
  def _get_vec_at_col(m, j, i)
    vec = LibGsl.gsl_vector_alloc(i)
    LibGsl.gsl_matrix_get_col(vec, m, j)
    return Vector.new(vec.value, i)
  end

  def _normalize_range(rng : Range, n)
    first = (rng.begin.nil? ? 0 : rng.begin).as(Int32)
    last = (rng.end.nil? ? n : rng.end).as(Int32)
    return (first...last)
  end

  def _slice_matrix_submatrix(m, k1, n1, k2, n2)
    view = LibGsl.gsl_matrix_submatrix(m, k1, k2, n1-k1, n2-k2)
    return Matrix(Float64).new(view.matrix, n1-k1, n2-k2)
  end

  def _transpose_mat(m, r, c)
    ptm = LibGsl.gsl_matrix_alloc(c, r)
    LibGsl.gsl_matrix_transpose_memcpy(ptm, m)
    return Matrix(Float64).new ptm.value, c, r
  end
end
