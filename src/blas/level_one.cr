require "../libs/dtype"
require "../libs/cblas"
require "../vector/*"

macro blas_helper(dtype, matrix_prefix, vector_prefix, matrix_type, vector_type, blas_prefix)
  module LL
    extend self

    def givens(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}), da, db, c, s)
      LibCblas.{{blas_prefix}}rotg(pointerof(da), pointerof(db), pointerof(c), pointerof(s))
      LibCblas.{{blas_prefix}}rot(a.size, a.data, a.stride, b.data, b.stride, pointerof(c), pointerof(s))
      return a, b
    end

    # Computes the dot product of two vectors
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [4.0, 5.0, 6.0]
    # v1.dot(v2) # => 32
    # ```
    def dot(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}))
      LibCblas.{{blas_prefix}}dot(a.size, a.data, a.stride, b.data, b.stride)
    end

    # Computes the euclidean norm of the vector
    #
    # ```
    # vec = Vector.new [1.0, 2.0, 3.0]
    # vec.norm # => 3.741657386773941
    # ```
    def norm(a : Vector({{vector_type}}, {{dtype}}))
      LibCblas.{{blas_prefix}}nrm2(a.size, a.data, a.stride)
    end

    # Sum of absolute values
    #
    # ```
    # v1 = Vector.new [-1, 1, 2]
    # v2.asum # => 4
    # ```
    def asum(a : Vector({{vector_type}}, {{dtype}}))
      LibCblas.{{blas_prefix}}asum(a.size, a.data, a.stride)
    end

    # Index of absolute value max
    #
    # ```
    # v1 = Vector.new [-8, 1, 2]
    # v2.amax # => 0
    # ```
    def amax(a : Vector({{vector_type}}, {{dtype}}))
      LibCblas.i{{blas_prefix}}amax(a.size, a.data, a.stride)
    end

  end
end

blas_helper Float64, gsl_matrix, gsl_vector, LibGsl::GslMatrix, LibGsl::GslVector, d
blas_helper Float32, gsl_matrix_float, gsl_vector_float, LibGsl::GslMatrixFloat, LibGsl::GslVectorFloat, s
