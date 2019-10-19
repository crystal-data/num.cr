require "../libs/dtype"
require "../libs/gsl"
require "../vector/*"

macro stats_helper(dtype, matrix_prefix, vector_prefix, matrix_type, vector_type)
  module LL
    extend self
    # Computes the maximum value of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.max # => 4
    # ```
    def max(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_max(v.ptr)
    end

    # Computes the minimum value of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.min # => 1
    # ```
    def min(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_min(v.ptr)
    end

    # Computes the min and max values of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.ptpv # => {1, 4}
    # ```
    def ptpv(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_minmax(v.ptr, out min_out, out max_out)
      return min_out, max_out
    end

    # Computes the "peak to peak" of a vector (max - min)
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.ptp # => 3
    # ```
    def ptpv(v : Vector({{vector_type}}, {{dtype}}))
      mn, mx = ptpv(v)
      return mx-mn
    end

    # Computes the index of the maximum value of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.argmax # => 3
    # ```
    def argmax(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_max_index(v.ptr)
    end

    # Computes the index of the minimum value of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.argmin # => 0
    # ```
    def argmin(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_min_index(v.ptr)
    end

    # Computes the indexes of the minimum and maximum values of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.argminmax # => {0, 3}
    # ```
    def argminmax(v : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_minmax_index(v.ptr, out imin, out imax)
      return imin, imax
    end

    def cumsum(v : Vector({{vector_type}}, {{dtype}}))
      (1...v.size).each do |index|
        v[index] += v[index - 1]
      end
      return v
    end

    # Computes the cumulative product of a vector
    #
    # ```
    # v = Vector.new [1, 2, 3, 4]
    # v.cumprod # => [1, 2, 6, 24]
    # ```
    def cumprod(v : Vector({{vector_type}}, {{dtype}}))
      (1...v.size).each do |index|
        v[index] *= v[index - 1]
      end
      return v
    end
  end
end

stats_helper Float64, gsl_matrix, gsl_vector, LibGsl::GslMatrix, LibGsl::GslVector
stats_helper Float32, gsl_matrix_float, gsl_vector_float, LibGsl::GslMatrixFloat, LibGsl::GslVectorFloat
