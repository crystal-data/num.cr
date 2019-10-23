require "../libs/dtype"
require "../libs/cblas"
require "../core/vector"

module Bottle
  macro blas_helper(dtype, blas_prefix, cast)
    module Bottle::BLAS
      extend self

      # Forms the dot product of two Flasks.
      # Uses unrolled loops for increments equal to one.
      #
      # ```
      # v1 = Vector.new [1.0, 2.0, 3.0]
      # v2 = Vector.new [4.0, 5.0, 6.0]
      # dot(v1, v2) # => 32
      # ```
      def dot(a : Vector({{dtype}}), b : Vector({{dtype}}))
        LibCblas.{{blas_prefix}}dot(a.size, a.data, a.stride, b.data, b.stride)
      end

      def axpy(a : Vector({{dtype}}), b : Vector({{dtype}}), alpha = 1{{cast}})
        LibCblas.{{blas_prefix}}axpy(a.size, alpha, a.data, a.stride, b.data, b.stride)
        b
      end

      # returns the euclidean norm of a Flask so that
      # `norm := sqrt( x'*x )`
      #
      # ```
      # vec = Vector.new [1.0, 2.0, 3.0]
      # norm(vec) # => 3.741657386773941
      # ```
      def norm(a : Vector({{dtype}}))
        LibCblas.{{blas_prefix}}nrm2(a.size, a.data, a.stride)
      end

      # Takes the sum of the absolute values.
      # Uses unrolled loops for increment equal to one.
      #
      # ```
      # v1 = Vector.new [-1, 1, 2]
      # asum(v1) # => 4
      # ```
      def asum(a : Vector({{dtype}}))
        LibCblas.{{blas_prefix}}asum(a.size, a.data, a.stride)
      end

      # Finds the index of the first element having maximum
      # absolute value.
      #
      # ```
      # v1 = Vector.new [-8, 1, 2]
      # amax(v1) # => 0
      # ```
      def amax(a : Vector({{dtype}}))
        LibCblas.i{{blas_prefix}}amax(a.size, a.data, a.stride)
      end

      # Scales a vector by a constant.
      # Uses unrolled loops for increment equal to 1.
      #
      # ```
      # v1 = Vector.new [1, 2, 3]
      # scale(v1, 3) #=> [3, 6, 9]
      # ```
      def scale(a : Vector({{dtype}}), x : {{dtype}})
        LibCblas.{{blas_prefix}}scal(a.size, x, a.data, a.stride)
        return a
      end

    end
  end

  blas_helper Float64, d, _f64
  blas_helper Float32, s, _f32
end
