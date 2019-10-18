require "../libs/dtype"
require "../libs/gsl"
require "../vector2/*"

macro math_helper(dtype, matrix_prefix, vector_prefix, matrix_type, vector_type)
  module LL
    extend self

    # Elementwise addition of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 + v2 # => [3.0, 6.0, 9.0]
    # ```
    def add(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_add(a.ptr, b.ptr)
      return a
    end

    # Elementwise addition of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 + v2 # => [3.0, 4.0, 5.0]
    # ```
    def add(a : Vector({{vector_type}}, {{dtype}}), x)
      LibGsl.{{vector_prefix}}_add_constant(a.ptr, x)
      return a
    end

    # Elementwise subtraction of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 - v2 # => [-1.0, -2.0, -3.0]
    # ```
    def sub(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_sub(a.ptr, b.ptr)
      return a
    end

    # Elementwise subtraction of a vector with a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 - v2 # => [-1.0, 0.0, 1.0]
    # ```
    def sub(a : Vector({{vector_type}}, {{dtype}}), x)
      LibGsl.{{vector_prefix}}_add_constant(a.ptr, -x)
      return a
    end


      # Elementwise multiplication of a vector to another equally sized vector
      #
      # ```
      # v1 = Vector.new [1.0, 2.0, 3.0]
      # v2 = Vector.new [2.0, 4.0, 6.0]
      # v1 * v2 # => [3.0, 8.0, 18.0]
      # ```
    def mul(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_mul(a.ptr, b.ptr)
      return a
    end

    # Elementwise multiplication of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 + v2 # => [2.0, 4.0, 6.0]
    # ```
    def mul(a : Vector({{vector_type}}, {{dtype}}), x)
      LibGsl.{{vector_prefix}}_scale(a.ptr, x)
      return a
    end

    # Elementwise division of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 / v2 # => [0.5, 0.5, 0.5]
    # ```
    def div(a : Vector({{vector_type}}, {{dtype}}), b : Vector({{vector_type}}, {{dtype}}))
      LibGsl.{{vector_prefix}}_div(a.ptr, b.ptr)
      return a
    end

    # Elementwise division of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 / v2 # => [0.5, 1, 1.5]
    # ```
    def div(a : Vector({{vector_type}}, {{dtype}}), x)
      LibGsl.{{vector_prefix}}_scale(a.ptr, 1/x)
      return a
    end
  end
end

math_helper Float64, gsl_matrix, gsl_vector, LibGsl::GslMatrix, LibGsl::GslVector
math_helper Float32, gsl_matrix_float, gsl_vector_float, LibGsl::GslMatrixFloat, LibGsl::GslVectorFloat
