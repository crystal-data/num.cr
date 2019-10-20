require "./dtype"

@[Link("gsl")]
lib LibGsl
  # Blocks
  struct GslBlock
    size : UInt
    data : Double*
  end

  struct GslBlockFloat
    size : UInt
    data : Real*
  end

  fun gsl_block_alloc(n : UInt) : GslBlock*
  fun gsl_block_calloc(n : UInt) : GslBlock*
  fun gsl_block_free(b : GslBlock*)

  fun gsl_block_float_alloc(n : UInt) : GslBlockFloat*
  fun gsl_block_float_calloc(n : UInt) : GslBlockFloat*
  fun gsl_block_float_free(b : GslBlockFloat*)

  # Vector Types
  struct GslVector
    size : UInt
    stride : UInt
    data : Double*
    block : GslBlock*
    owner : Integer
  end

  struct GslVectorFloat
    size : UInt
    stride : UInt
    data : Real*
    block : GslBlockFloat*
    owner : Integer
  end

  # Vector Views
  struct GslVectorView
    vector : GslVector
  end

  struct GslVectorFloatView
    vector : GslVectorFloat
  end

  fun gsl_vector_alloc(n : UInt) : GslVector*
  fun gsl_vector_calloc(n : UInt) : GslVector*
  fun gsl_vector_free(b : GslVector*)
  fun gsl_vector_get(v : GslVector*, i : UInt) : Double
  fun gsl_vector_set(v : GslVector*, i : UInt, x : Double)
  fun gsl_vector_ptr(v : GslVector*, i : UInt) : Double*
  fun gsl_vector_set_all(v : GslVector*, x : Double)
  fun gsl_vector_set_zero(v : GslVector*)
  fun gsl_vector_set_basis(v : GslVector*, i : UInt)
  fun gsl_vector_subvector(v : GslVector*, offset : UInt, n : UInt) : GslVectorView
  fun gsl_vector_subvector_with_stride(v : GslVector*, offset : UInt, stride : UInt, n : UInt) : GslVectorView
  fun gsl_vector_view_array(base : Double*, n : UInt) : GslVectorView
  fun gsl_vector_view_array_with_stride(base : Double*, stride : UInt, n : UInt)
  fun gsl_vector_memcpy(dest : GslVector*, src : GslVector*) : Integer
  fun gsl_vector_swap_elements(v : GslVector*, i : UInt, j : UInt) : Integer
  fun gsl_vector_reverse(v : GslVector*) : Integer
  fun gsl_vector_add(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_sub(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_mul(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_div(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_scale(a : GslVector*, x : Double) : Integer
  fun gsl_vector_add_constant(a : GslVector*, x : Double)
  fun gsl_vector_axpby(alpha : Double, x : GslVector*, beta : Double, y : GslVector*)
  fun gsl_vector_max(v : GslVector*) : Double
  fun gsl_vector_min(v : GslVector*) : Double
  fun gsl_vector_minmax(v : GslVector*, min_out : Double*, max_out : Double*)
  fun gsl_vector_max_index(v : GslVector*) : UInt
  fun gsl_vector_min_index(v : GslVector*) : UInt
  fun gsl_vector_minmax_index(v : GslVector*, imin : UInt*, imax : UInt*)
  fun gsl_vector_isnull(v : GslVector*) : Integer
  fun gsl_vector_ispos(v : GslVector*) : Integer
  fun gsl_vector_isneg(v : GslVector*) : Integer
  fun gsl_vector_isnonneg(v : GslVector*) : Integer
  fun gsl_vector_equal(u : GslVector*, v : GslVector*) : Integer

  fun gsl_vector_float_alloc(n : UInt) : GslVectorFloat*
  fun gsl_vector_float_calloc(n : UInt) : GslVectorFloat*
  fun gsl_vector_float_free(b : GslVectorFloat*)
  fun gsl_vector_float_get(v : GslVectorFloat*, i : UInt) : Real
  fun gsl_vector_float_set(v : GslVectorFloat*, i : UInt, x : Real)
  fun gsl_vector_float_ptr(v : GslVectorFloat*, i : UInt) : Real*
  fun gsl_vector_float_set_all(v : GslVectorFloat*, x : Real)
  fun gsl_vector_float_set_zero(v : GslVectorFloat*)
  fun gsl_vector_float_set_basis(v : GslVectorFloat*, i : UInt)
  fun gsl_vector_float_subvector(v : GslVectorFloat*, offset : UInt, n : UInt) : GslVectorFloatView
  fun gsl_vector_float_subvector_with_stride(v : GslVectorFloat*, offset : UInt, stride : UInt, n : UInt) : GslVectorFloatView
  fun gsl_vector_float_view_array(base : Real*, n : UInt) : GslVectorFloatView
  fun gsl_vector_float_view_array_with_stride(base : Real*, stride : UInt, n : UInt)
  fun gsl_vector_float_memcpy(dest : GslVectorFloat*, src : GslVectorFloat*) : Integer
  fun gsl_vector_float_swap_elements(v : GslVectorFloat*, i : UInt, j : UInt) : Integer
  fun gsl_vector_float_reverse(v : GslVectorFloat*) : Integer
  fun gsl_vector_float_add(a : GslVectorFloat*, b : GslVectorFloat*) : Integer
  fun gsl_vector_float_sub(a : GslVectorFloat*, b : GslVectorFloat*) : Integer
  fun gsl_vector_float_mul(a : GslVectorFloat*, b : GslVectorFloat*) : Integer
  fun gsl_vector_float_div(a : GslVectorFloat*, b : GslVectorFloat*) : Integer
  fun gsl_vector_float_scale(a : GslVectorFloat*, x : Real) : Integer
  fun gsl_vector_float_add_constant(a : GslVectorFloat*, x : Real)
  fun gsl_vector_float_axpby(alpha : Real, x : GslVectorFloat*, beta : Real, y : GslVectorFloat*)
  fun gsl_vector_float_max(v : GslVectorFloat*) : Real
  fun gsl_vector_float_min(v : GslVectorFloat*) : Real
  fun gsl_vector_float_minmax(v : GslVectorFloat*, min_out : Real*, max_out : Real*)
  fun gsl_vector_float_max_index(v : GslVectorFloat*) : UInt
  fun gsl_vector_float_min_index(v : GslVectorFloat*) : UInt
  fun gsl_vector_float_minmax_index(v : GslVectorFloat*, imin : UInt*, imax : UInt*)
  fun gsl_vector_float_isnull(v : GslVectorFloat*) : Integer
  fun gsl_vector_float_ispos(v : GslVectorFloat*) : Integer
  fun gsl_vector_float_isneg(v : GslVectorFloat*) : Integer
  fun gsl_vector_float_isnonneg(v : GslVectorFloat*) : Integer
  fun gsl_vector_float_equal(u : GslVectorFloat*, v : GslVectorFloat*) : Integer

  # Matrix Types
  struct GslMatrix
    size1 : UInt
    size2 : UInt
    tda : UInt
    data : Double*
    block : GslBlock*
    owner : Integer
  end

  struct GslMatrixFloat
    size1 : UInt
    size2 : UInt
    tda : UInt
    data : Real*
    block : GslBlockFloat*
    owner : Integer
  end

  # Matrix Views
  struct GslMatrixView
    vector : GslMatrix
  end

  struct GslMatrixFloatView
    vector : GslMatrixFloat
  end

  fun gsl_matrix_alloc(n1 : UInt, n2 : UInt) : GslMatrix*
  fun gsl_matrix_allow(n1 : UInt, n2 : UInt) : GslMatrix*
  fun gsl_matrix_free(m : GslMatrix*)
  fun gsl_matrix_get(m : GslMatrix*, i : UInt, j : UInt) : Double
  fun gsl_matrix_set(m : GslMatrix*, i : UInt, j : UInt, x : Double)
  fun gsl_matrix_set_all(m : GslMatrix*, x : Double)
  fun gsl_matrix_set_zero(m : GslMatrix*)
  fun gsl_matrix_set_identify(m : GslMatrix*)
  fun gsl_matrix_get_row(v : GslVector*, m : GslMatrix*, i : UInt) : Integer
  fun gsl_matrix_get_col(v : GslVector*, m : GslMatrix*, i : UInt) : Integer
  fun gsl_matrix_set_row(m : GslMatrix*, i : UInt, v : GslVector*) : Integer
  fun gsl_matrix_set_col(m : GslMatrix*, i : UInt, v : GslVector*) : Integer
  fun gsl_matrix_submatrix(m : GslMatrix*, k1 : UInt, k2 : UInt, n1 : UInt, n2 : UInt) : GslMatrixView
  fun gsl_matrix_row(m : GslMatrix*, i : UInt) : GslVectorView
  fun gsl_matrix_column(m : GslMatrix*, j : UInt) : GslVectorView
  fun gsl_matrix_subrow(m : GslMatrix*, i : UInt, offset : UInt, n : UInt) : GslVectorView
  fun gsl_matrix_subcolumn(m : GslMatrix*, i : UInt, offset : UInt, n : UInt) : GslVectorView
  fun gsl_matrix_diagonal(m : GslMatrix*) : GslVectorView
  fun gsl_matrix_subdiagonal(m : GslMatrix*, k : UInt) : GslVectorView
  fun gsl_matrix_superdiagonal(m : GslMatrix*, k : UInt)
  fun gsl_matrix_swap_rows(m : GslMatrix*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_swap_columns(m : GslMatrix*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_swap_rowcol(m : GslMatrix*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_transpose_memcpy(dest : GslMatrix*, src : GslMatrix*) : Integer
  fun gsl_matrix_transpose(m : GslMatrix*) : Integer
  fun gsl_matrix_memcpy(dest : GslMatrix*, src : GslMatrix*)
  fun gsl_matrix_add(a : GslMatrix*, b : GslMatrix*) : Integer
  fun gsl_matrix_sub(a : GslMatrix*, b : GslMatrix*)
  fun gsl_matrix_mul_elements(a : GslMatrix*, b : GslMatrix*) : Integer
  fun gsl_matrix_div_elements(a : GslMatrix*, b : GslMatrix*) : Integer
  fun gsl_matrix_scale(a : GslMatrix*, x : Double) : Integer
  fun gsl_matrix_add_constant(a : GslMatrix*, x : Double)
  fun gsl_matrix_max(m : GslMatrix*) : Double
  fun gsl_matrix_min(m : GslMatrix*) : Double
  fun gsl_matrix_minmax(m : GslMatrix*, min_out : Double*, max_out : Double*)
  fun gsl_matrix_max_index(m : GslMatrix*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_min_index(m : GslMatrix*, imin : UInt*, jmin : UInt*)
  fun gsl_matrix_minmax_index(m : GslMatrix*, imin : UInt*, jmin : UInt*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_isnull(m : GslMatrix*) : Integer
  fun gsl_matrix_ispos(m : GslMatrix*) : Integer
  fun gsl_matrix_isneg(m : GslMatrix*) : Integer
  fun gsl_matrix_isnonneg(m : GslMatrix*) : Integer

  fun gsl_matrix_float_alloc(n1 : UInt, n2 : UInt) : GslMatrixFloat*
  fun gsl_matrix_float_allow(n1 : UInt, n2 : UInt) : GslMatrixFloat*
  fun gsl_matrix_float_free(m : GslMatrixFloat*)
  fun gsl_matrix_float_get(m : GslMatrixFloat*, i : UInt, j : UInt) : Real
  fun gsl_matrix_float_set(m : GslMatrixFloat*, i : UInt, j : UInt, x : Real)
  fun gsl_matrix_float_set_all(m : GslMatrixFloat*, x : Real)
  fun gsl_matrix_float_set_zero(m : GslMatrixFloat*)
  fun gsl_matrix_float_set_identify(m : GslMatrixFloat*)
  fun gsl_matrix_float_get_row(v : GslVectorFloat*, m : GslMatrixFloat*, i : UInt) : Integer
  fun gsl_matrix_float_get_col(v : GslVectorFloat*, m : GslMatrixFloat*, i : UInt) : Integer
  fun gsl_matrix_float_set_row(m : GslMatrixFloat*, i : UInt, v : GslVectorFloat*) : Integer
  fun gsl_matrix_float_set_col(m : GslMatrixFloat*, i : UInt, v : GslVectorFloat*) : Integer
  fun gsl_matrix_float_submatrix(m : GslMatrixFloat*, k1 : UInt, k2 : UInt, n1 : UInt, n2 : UInt) : GslMatrixFloatView
  fun gsl_matrix_float_row(m : GslMatrixFloat*, i : UInt) : GslVectorFloatView
  fun gsl_matrix_float_column(m : GslMatrixFloat*, j : UInt) : GslVectorFloatView
  fun gsl_matrix_float_subrow(m : GslMatrixFloat*, i : UInt, offset : UInt, n : UInt) : GslVectorFloatView
  fun gsl_matrix_float_subcolumn(m : GslMatrixFloat*, i : UInt, offset : UInt, n : UInt) : GslVectorFloatView
  fun gsl_matrix_float_diagonal(m : GslMatrixFloat*) : GslVectorFloatView
  fun gsl_matrix_float_subdiagonal(m : GslMatrixFloat*, k : UInt) : GslVectorFloatView
  fun gsl_matrix_float_superdiagonal(m : GslMatrixFloat*, k : UInt)
  fun gsl_matrix_float_swap_rows(m : GslMatrixFloat*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_float_swap_columns(m : GslMatrixFloat*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_float_swap_rowcol(m : GslMatrixFloat*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_float_transpose_memcpy(dest : GslMatrixFloat*, src : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_transpose(m : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_memcpy(dest : GslMatrixFloat*, src : GslMatrixFloat*)
  fun gsl_matrix_float_add(a : GslMatrixFloat*, b : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_sub(a : GslMatrixFloat*, b : GslMatrixFloat*)
  fun gsl_matrix_float_mul_elements(a : GslMatrixFloat*, b : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_div_elements(a : GslMatrixFloat*, b : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_scale(a : GslMatrixFloat*, x : Real) : Integer
  fun gsl_matrix_float_add_constant(a : GslMatrixFloat*, x : Real)
  fun gsl_matrix_float_max(m : GslMatrixFloat*) : Real
  fun gsl_matrix_float_min(m : GslMatrixFloat*) : Real
  fun gsl_matrix_float_minmax(m : GslMatrixFloat*, min_out : Real*, max_out : Real*)
  fun gsl_matrix_float_max_index(m : GslMatrixFloat*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_float_min_index(m : GslMatrixFloat*, imin : UInt*, jmin : UInt*)
  fun gsl_matrix_float_minmax_index(m : GslMatrixFloat*, imin : UInt*, jmin : UInt*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_float_isnull(m : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_ispos(m : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_isneg(m : GslMatrixFloat*) : Integer
  fun gsl_matrix_float_isnonneg(m : GslMatrixFloat*) : Integer
end
