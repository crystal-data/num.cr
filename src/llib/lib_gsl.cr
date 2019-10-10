@[Link("gsl")]
@[Link("openblas")]
lib LibGsl
  alias Integer = LibC::Int
  alias Real = LibC::Float
  alias Double = LibC::Double
  alias Logical = LibC::Char
  alias Ftnlen = LibC::Int
  alias LFp = Pointer(Void)
  alias UInt = LibC::SizeT

  struct GslBlock
    size : UInt*
    data : Double*
  end

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

  # Block allocation
  fun gsl_block_alloc(n : UInt) : GslBlock
  fun gsl_block_calloc(n : UInt) : GslBlock
  fun gsl_block_free(b : GslBlock)

  # Vector allocation
  fun gsl_vector_alloc(n : UInt) : GslVector*
  fun gsl_vector_calloc(n : UInt) : GslVector*
  fun gsl_vector_free(b : GslVector*)

  # Vector getters/setters
  fun gsl_vector_get(v : GslVector*, i : UInt) : Double
  fun gsl_vector_set(v : GslVector*, i : UInt, x : Double)
  fun gsl_vector_set_all(v : GslVector*, x: Double)
  fun gsl_vector_set_zero(v : GslVector*)
  fun gsl_vector_set_basis(v : GslVector*, i : UInt)

  # Vector views
  fun gsl_vector_subvector(v : GslVector*, offset : UInt, n : UInt) : GslVectorView
  fun gsl_vector_subvector_with_stride(v : GslVector*, offset : UInt, stride : UInt, n : UInt) : GslVectorView
  fun gsl_vector_view_array(base : Double*, n : UInt) : GslVectorView
  fun gsl_vector_view_array_with_stride(base : Double*, stride : UInt, n : UInt)

  # Vector manipulations
  fun gsl_vector_memcpy(dest : GslVector*, src : GslVector*) : Integer
  fun gsl_vector_swap_elements(v : GslVector*, i : UInt, j : UInt) : Integer
  fun gsl_vector_reverse(v : GslVector*) : Integer

  # Vector math
  fun gsl_vector_add(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_sub(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_mul(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_div(a : GslVector*, b : GslVector*) : Integer
  fun gsl_vector_scale(a : GslVector*, x : Double) : Integer
  fun gsl_vector_add_constant(a : GslVector*, x : Double)
  fun gsl_vector_axpby(alpha : Double, x : GslVector*, beta : Double, y : GslVector*)

  # Vector max/min
  fun gsl_vector_max(v : GslVector*) : Double
  fun gsl_vector_min(v : GslVector*) : Double
  fun gsl_vector_minmax(v : GslVector*, min_out : Double*, max_out : Double*)
  fun gsl_vector_max_index(v : GslVector*) : UInt
  fun gsl_vector_min_index(v : GslVector*) : UInt
  fun gsl_vector_minmax_index(v : GslVector*, imin : UInt*, imax : UInt*)

  # properties
  fun gsl_vector_isnull(v : GslVector*) : Integer
  fun gsl_vector_ispos(v : GslVector*) : Integer
  fun gsl_vector_isneg(v : GslVector*) : Integer
  fun gsl_vector_isnonneg(v : GslVector*) : Integer
  fun gsl_vector_equal(u : GslVector*, v : GslVector*) : Integer

  # Blas absmin
  fun gsl_blas_idamax(x : GslVector*) : Integer
end
