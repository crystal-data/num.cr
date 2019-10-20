require "../libs/dtype"
require "../libs/cblas"
require "../flask/*"

macro blas_helper(dtype, blas_prefix, cast)
  module LL
    extend self

    # Computes the dot product of two vectors
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [4.0, 5.0, 6.0]
    # v1.dot(v2) # => 32
    # ```
    def dot(a : Flask({{dtype}}), b : Flask({{dtype}}))
      LibCblas.{{blas_prefix}}dot(a.size, a.data, a.stride, b.data, b.stride)
    end

    # Computes the euclidean norm of the vector
    #
    # ```
    # vec = Vector.new [1.0, 2.0, 3.0]
    # vec.norm # => 3.741657386773941
    # ```
    def norm(a : Flask({{dtype}}))
      LibCblas.{{blas_prefix}}nrm2(a.size, a.data, a.stride)
    end

    # Sum of absolute values
    #
    # ```
    # v1 = Vector.new [-1, 1, 2]
    # v2.asum # => 4
    # ```
    def asum(a : Flask({{dtype}}))
      LibCblas.{{blas_prefix}}asum(a.size, a.data, a.stride)
    end

    # Index of absolute value max
    #
    # ```
    # v1 = Vector.new [-8, 1, 2]
    # v2.amax # => 0
    # ```
    def amax(a : Flask({{dtype}}))
      LibCblas.i{{blas_prefix}}amax(a.size, a.data, a.stride)
    end

    def scale(a : Flask({{dtype}}), x : {{dtype}})
      LibCblas.{{blas_prefix}}scal(a.size, x, a.data, a.stride)
      return a
    end

  end
end

blas_helper Float64, d, _f64
blas_helper Float32, s, _f32
