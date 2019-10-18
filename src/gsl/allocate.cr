require "../libs/dtype"
require "../libs/gsl"
require "complex"

macro allocation_helper(dtype, matrix_prefix, vector_prefix, matrix_type, vector_type)
  module LL
    extend self
    def allocate(typ : {{dtype}}.class, otyp : {{vector_type}}.class, n : Number)
      LibGsl.{{vector_prefix}}_alloc(n)
    end

    def allocate(typ : {{dtype}}.class, otyp : {{matrix_type}}.class, n1 : Number, n2 : Number)
      LibGsl.{{matrix_prefix}}_alloc(n1, n2)
    end

    def zero(typ : {{dtype}}.class, otyp : {{vector_type}}.class, n : Number)
      LibGsl.{{vector_prefix}}_calloc(n)
    end

    def zero(typ : {{dtype}}.class, otyp : {{matrix_type}}.class, n1 : Number, n2 : Number)
      LibGsl.{{matrix_prefix}}_calloc(n1, n2)
    end

    def free(v : Pointer({{vector_type}}))
      LibGsl.{{vector_prefix}}_free(v)
    end

    def free(m : Pointer({{matrix_type}}))
      LibGsl.{{matrix_prefix}}_free(m)
    end

    def memcpy(dest : Pointer({{vector_type}}), src : Pointer({{vector_type}}))
      LibGsl.{{vector_prefix}}_memcpy(dest, src)
    end

    def reverse(v : Pointer({{vector_type}}))
      LibGsl.{{vector_prefix}}_reverse(v)
    end
  end
end

allocation_helper Float64, gsl_matrix, gsl_vector, LibGsl::GslMatrix, LibGsl::GslVector
allocation_helper Float32, gsl_matrix_float, gsl_vector_float, LibGsl::GslMatrixFloat, LibGsl::GslVectorFloat
