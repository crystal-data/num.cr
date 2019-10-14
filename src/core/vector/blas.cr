require "../../vector/*"
require "../../libs/gsl"
require "../../libs/cblas"

module Bottle::Core::VectorBlas
  extend self

  # Computes the dot product of two vectors
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [4.0, 5.0, 6.0]
  # v1.dot(v2) # => 32
  # ```
  def vector_dot(vector : Vector(LibGsl::GslVector, Float64), other : Vector(LibGsl::GslVector, Float64))
    LibCblas.ddot(vector.size, vector.data, vector.stride, other.data, other.stride)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Vector.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def vector_norm(vector : Vector(LibGsl::GslVector, Float64))
    LibCblas.dnrm2(vector.size, vector.data, vector.stride)
  end

  # Elementwise multiplication of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 + v2 # => [2.0, 4.0, 6.0]
  # ```
  def vector_scale(vector : Vector(LibGsl::GslVector, Float64), alpha : Number)
    LibCblas.dscal(vector.size, alpha, vector.data, vector.stride)
    return vector
  end

  # Sum of absolute values
  #
  # ```
  # v1 = Vector.new [-1, 1, 2]
  # v2.asum # => 4
  # ```
  def vector_absolute_sum(vector : Vector(LibGsl::GslVector, Float64))
    LibCblas.dasum(vector.size, vector.data, vector.stride)
  end

  # Index of absolute value max
  #
  # ```
  # v1 = Vector.new [-8, 1, 2]
  # v2.amax # => 0
  # ```
  def vector_absolute_max(vector : Vector(LibGsl::GslVector, Float64))
    LibCblas.idamax(vector.size, vector.data, vector.stride)
  end
end
