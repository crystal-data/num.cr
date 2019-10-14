@[Link("gsl")]
lib LibGsl
  alias Integer = LibC::Int
  alias Real = LibC::Float
  alias Double = LibC::Double
  alias Logical = LibC::Char
  alias Ftnlen = LibC::Int
  alias LFp = Pointer(Void)
  alias UInt = LibC::SizeT

  # ###############DOUBLE#####################
  struct GslBlock
    size : UInt
    data : Double*
  end

  fun gsl_block_alloc(n : UInt) : GslBlock*
  fun gsl_block_calloc(n : UInt) : GslBlock*
  fun gsl_block_free(b : GslBlock*)

  struct GslVector
    size : UInt
    stride : UInt
    data : Double*
    block : GslBlock*
    owner : Integer
  end

  struct GslVectorView
    vector : GslVector
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

  # Blas Level 1
  fun gsl_blas_ddot(x : GslVector*, y : GslVector*, result : Double*)
  fun gsl_blas_dnrm2(x : GslVector*) : Double
  fun gsl_blas_dasum(x : GslVector*) : Double
  fun gsl_blas_idamax(x : GslVector*) : UInt
  fun gsl_blas_daxpy(alpha : Double, x : GslVector*, y : GslVector*)

  struct GslMatrix
    size1 : UInt
    size2 : UInt
    tda : UInt
    data : Double*
    block : GslBlock*
    owner : Integer
  end

  struct GslMatrixView
    matrix : GslMatrix
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

  # ###############DOUBLE#####################

  # ###############INTEGER#####################
  struct GslBlockInt
    size : UInt
    data : Integer*
  end

  fun gsl_block_int_alloc(n : UInt) : GslBlockInt*
  fun gsl_block_int_calloc(n : UInt) : GslBlockInt*
  fun gsl_block_int_free(b : GslBlockInt*)

  struct GslVectorInt
    size : UInt
    stride : UInt
    data : Integer*
    block : GslBlockInt*
    owner : Integer
  end

  struct GslVectorIntView
    vector : GslVectorInt
  end

  fun gsl_vector_int_alloc(n : UInt) : GslVectorInt*
  fun gsl_vector_int_calloc(n : UInt) : GslVectorInt*
  fun gsl_vector_int_free(b : GslVectorInt*)
  fun gsl_vector_int_get(v : GslVectorInt*, i : UInt) : Integer
  fun gsl_vector_int_set(v : GslVectorInt*, i : UInt, x : Integer)
  fun gsl_vector_int_ptr(v : GslVectorInt*, i : UInt) : Double*
  fun gsl_vector_int_set_all(v : GslVectorInt*, x : Integer)
  fun gsl_vector_int_set_zero(v : GslVectorInt*)
  fun gsl_vector_int_set_basis(v : GslVectorInt*, i : UInt)
  fun gsl_vector_int_subvector(v : GslVectorInt*, offset : UInt, n : UInt) : GslVectorIntView
  fun gsl_vector_int_subvector_with_stride(v : GslVectorInt*, offset : UInt, stride : UInt, n : UInt) : GslVectorIntView
  fun gsl_vector_int_view_array(base : Integer*, n : UInt) : GslVectorIntView
  fun gsl_vector_int_view_array_with_stride(base : Integer*, stride : UInt, n : UInt)
  fun gsl_vector_int_memcpy(dest : GslVectorInt*, src : GslVectorInt*) : Integer
  fun gsl_vector_int_swap_elements(v : GslVectorInt*, i : UInt, j : UInt) : Integer
  fun gsl_vector_int_reverse(v : GslVectorInt*) : Integer
  fun gsl_vector_int_add(a : GslVectorInt*, b : GslVectorInt*) : Integer
  fun gsl_vector_int_sub(a : GslVectorInt*, b : GslVectorInt*) : Integer
  fun gsl_vector_int_mul(a : GslVectorInt*, b : GslVectorInt*) : Integer
  fun gsl_vector_int_div(a : GslVectorInt*, b : GslVectorInt*) : Integer
  fun gsl_vector_int_scale(a : GslVectorInt*, x : Integer) : Integer
  fun gsl_vector_int_add_constant(a : GslVectorInt*, x : Integer)
  fun gsl_vector_int_axpby(alpha : Double, x : GslVectorInt*, beta : Integer, y : GslVectorInt*)
  fun gsl_vector_int_max(v : GslVectorInt*) : Integer
  fun gsl_vector_int_min(v : GslVectorInt*) : Integer
  fun gsl_vector_int_minmax(v : GslVectorInt*, min_out : Integer*, max_out : Integer*)
  fun gsl_vector_int_max_index(v : GslVectorInt*) : UInt
  fun gsl_vector_int_min_index(v : GslVectorInt*) : UInt
  fun gsl_vector_int_minmax_index(v : GslVectorInt*, imin : UInt*, imax : UInt*)
  fun gsl_vector_int_isnull(v : GslVectorInt*) : Integer
  fun gsl_vector_int_ispos(v : GslVectorInt*) : Integer
  fun gsl_vector_int_isneg(v : GslVectorInt*) : Integer
  fun gsl_vector_int_isnonneg(v : GslVectorInt*) : Integer
  fun gsl_vector_int_equal(u : GslVectorInt*, v : GslVectorInt*) : Integer

  struct GslMatrixInt
    size1 : UInt
    size2 : UInt
    tda : UInt
    data : Integer*
    block : GslBlockInt*
    owner : Integer
  end

  struct GslMatrixIntView
    matrix : GslMatrixInt
  end

  fun gsl_matrix_int_alloc(n1 : UInt, n2 : UInt) : GslMatrixInt*
  fun gsl_matrix_int_allow(n1 : UInt, n2 : UInt) : GslMatrixInt*
  fun gsl_matrix_int_free(m : GslMatrixInt*)
  fun gsl_matrix_int_get(m : GslMatrixInt*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_int_set(m : GslMatrixInt*, i : UInt, j : UInt, x : Integer)
  fun gsl_matrix_int_set_all(m : GslMatrixInt*, x : Integer)
  fun gsl_matrix_int_set_zero(m : GslMatrixInt*)
  fun gsl_matrix_int_set_identify(m : GslMatrixInt*)
  fun gsl_matrix_int_get_row(v : GslVectorInt*, m : GslMatrixInt*, i : UInt) : Integer
  fun gsl_matrix_int_get_col(v : GslVectorInt*, m : GslMatrixInt*, i : UInt) : Integer
  fun gsl_matrix_int_set_row(m : GslMatrixInt*, i : UInt, v : GslVectorInt*) : Integer
  fun gsl_matrix_int_set_col(m : GslMatrixInt*, i : UInt, v : GslVectorInt*) : Integer
  fun gsl_matrix_int_submatrix(m : GslMatrixInt*, k1 : UInt, k2 : UInt, n1 : UInt, n2 : UInt) : GslMatrixIntView
  fun gsl_matrix_int_row(m : GslMatrixInt*, i : UInt) : GslVectorIntView
  fun gsl_matrix_int_column(m : GslMatrixInt*, j : UInt) : GslVectorIntView
  fun gsl_matrix_int_subrow(m : GslMatrixInt*, i : UInt, offset : UInt, n : UInt) : GslVectorIntView
  fun gsl_matrix_int_subcolumn(m : GslMatrixInt*, i : UInt, offset : UInt, n : UInt) : GslVectorIntView
  fun gsl_matrix_int_diagonal(m : GslMatrixInt*) : GslVectorIntView
  fun gsl_matrix_int_subdiagonal(m : GslMatrixInt*, k : UInt) : GslVectorIntView
  fun gsl_matrix_int_superdiagonal(m : GslMatrixInt*, k : UInt)
  fun gsl_matrix_int_swap_rows(m : GslMatrixInt*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_int_swap_columns(m : GslMatrixInt*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_int_swap_rowcol(m : GslMatrixInt*, i : UInt, j : UInt) : Integer
  fun gsl_matrix_int_transpose_memcpy(dest : GslMatrixInt*, src : GslMatrixInt*) : Integer
  fun gsl_matrix_int_transpose(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_memcpy(dest : GslMatrixInt*, src : GslMatrixInt*)
  fun gsl_matrix_int_add(a : GslMatrixInt*, b : GslMatrixInt*) : Integer
  fun gsl_matrix_int_sub(a : GslMatrixInt*, b : GslMatrixInt*)
  fun gsl_matrix_int_mul_elements(a : GslMatrixInt*, b : GslMatrixInt*) : Integer
  fun gsl_matrix_int_div_elements(a : GslMatrixInt*, b : GslMatrixInt*) : Integer
  fun gsl_matrix_int_scale(a : GslMatrixInt*, x : Integer) : Integer
  fun gsl_matrix_int_add_constant(a : GslMatrixInt*, x : Integer)
  fun gsl_matrix_int_max(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_min(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_minmax(m : GslMatrixInt*, min_out : Integer*, max_out : Integer*)
  fun gsl_matrix_int_max_index(m : GslMatrixInt*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_int_min_index(m : GslMatrixInt*, imin : UInt*, jmin : UInt*)
  fun gsl_matrix_int_minmax_index(m : GslMatrixInt*, imin : UInt*, jmin : UInt*, imax : UInt*, jmax : UInt*)
  fun gsl_matrix_int_isnull(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_ispos(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_isneg(m : GslMatrixInt*) : Integer
  fun gsl_matrix_int_isnonneg(m : GslMatrixInt*) : Integer
  # ###############INTEGER#####################
end
