require "../llib/lib_gsl"
require "../core/vector"

module Bottle::Util::Arithmetic
  extend self

  # Vector arithmetic
  def _add_vec(a, b)
    LibGsl.gsl_vector_add(a.ptr, b.ptr)
    return a
  end

  def _add_vec_constant(a, x)
    LibGsl.gsl_vector_add_constant(a.ptr, x)
    return a
  end

  def _sub_vec(a, b)
    LibGsl.gsl_vector_sub(a.ptr, b.ptr)
    return a
  end

  def _sub_vec_constant(a, x)
    LibGsl.gsl_vector_add_constant(a.ptr, -x)
    return a
  end

  def _mul_vec(a, b)
    LibGsl.gsl_vector_mul(a.ptr, b.ptr)
    return a
  end

  def _mul_vec_constant(a, x)
    LibGsl.gsl_vector_scale(a.ptr, x)
    return a
  end

  def _div_vec(a, b)
    LibGsl.gsl_vector_scale(a.ptr, b.ptr)
    return a
  end

  def _div_vec_constant(a, x)
    LibGsl.gsl_vector_scale(a.ptr, 1 / x)
    return a
  end

  # Matrix arithmetic
  def _add_mat(a, b)
    LibGsl.gsl_matrix_add(a.ptr, b.ptr)
    return a
  end

  def _add_mat_constant(a, x)
    LibGsl.gsl_matrix_add_constant(a.ptr, x)
    return a
  end

  def _sub_mat(a, b)
    LibGsl.gsl_matrix_sub(a.ptr, b.ptr)
    return a
  end

  def _sub_mat_constant(a, x)
    LibGsl.gsl_matrix_add_constant(a.ptr, -x)
    return a
  end

  def _mul_mat(a, b)
    LibGsl.gsl_matrix_mul_elements(a.ptr, b.ptr)
    return a
  end

  def _mul_mat_constant(a, x)
    LibGsl.gsl_matrix_scale(a.ptr, x)
    return a
  end

  def _div_mat(a, b)
    LibGsl.gsl_matrix_div_elements(a.ptr, b.ptr)
    return a
  end

  def _div_mat_constant(a, x)
    LibGsl.gsl_matrix_scale(a.ptr, 1 / x)
    return a
  end
end
