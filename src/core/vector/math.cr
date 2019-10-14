require "../../libs/gsl"
require "../../vector/*"
require "../exceptions"

macro vector_arithmetic_abstract(type_, dtype, prefix)
  module Bottle::Core::VectorMath
    include Bottle::Core::Exceptions
    extend self

    # Elementwise addition of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 + v2 # => [3.0, 6.0, 9.0]
    # ```
    def add(a : Vector({{ type_ }}, {{ dtype }}), b : Vector)
      LibGsl.{{ prefix }}_add(a.ptr, b.ptr)
      return a
    end

    # Elementwise addition of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 + v2 # => [3.0, 4.0, 5.0]
    # ```
    def add_constant(a : Vector({{ type_ }}, {{ dtype }}), x : Number)
      LibGsl.{{ prefix }}_add_constant(a.ptr, x)
      return a
    end

    # Elementwise subtraction of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 - v2 # => [-1.0, -2.0, -3.0]
    # ```
    def sub(a : Vector({{ type_ }}, {{ dtype }}), b : Vector)
      LibGsl.{{ prefix }}_sub(a.ptr, b.ptr)
      return a
    end

    # Elementwise subtraction of a vector with a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 - v2 # => [-1.0, 0.0, 1.0]
    # ```
    def sub_constant(a : Vector({{ type_ }}, {{ dtype }}), x : Number)
      LibGsl.{{ prefix }}_add_constant(a.ptr, -x)
      return a
    end

    # Elementwise multiplication of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 * v2 # => [3.0, 8.0, 18.0]
    # ```
    def mul(a : Vector({{ type_ }}, {{ dtype }}), b : Vector)
      LibGsl.{{ prefix }}_mul(a.ptr, b.ptr)
      return a
    end

    # Elementwise multiplication of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 + v2 # => [2.0, 4.0, 6.0]
    # ```
    def mul_constant(a : Vector({{ type_ }}, {{ dtype }}), x : Number)
      LibGsl.{{ prefix }}_scale(a.ptr, x)
      return a
    end

    # Elementwise division of a vector to another equally sized vector
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = Vector.new [2.0, 4.0, 6.0]
    # v1 / v2 # => [0.5, 0.5, 0.5]
    # ```
    def div(a : Vector({{ type_ }}, {{ dtype }}), b : Vector)
      LibGsl.{{ prefix }}_div(a.ptr, b.ptr)
      return a
    end


    # Elementwise division of a vector to a scalar
    #
    # ```
    # v1 = Vector.new [1.0, 2.0, 3.0]
    # v2 = 2
    # v1 / v2 # => [0.5, 1, 1.5]
    # ```
    def div_constant(a : Vector({{ type_ }}, {{ dtype }}), x : Number)
      LibGsl.{{ prefix }}_scale(a.ptr, 1 / x)
      return a
    end

  end
end

vector_arithmetic_abstract LibGsl::GslVector, Float64, gsl_vector
vector_arithmetic_abstract LibGsl::GslVectorInt, Int32, gsl_vector_int
