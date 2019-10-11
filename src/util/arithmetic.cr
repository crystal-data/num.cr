require "../llib/gsl"
require "../core/vector"

macro arithmetic_abstract(type_, prefix)
  module Bottle::Util::VectorMath
    include Bottle::Util::Exceptions
    extend self

    def add(a : Vector({{ type_ }}), b : Vector)
      LibGsl.{{ prefix }}_add(a.ptr, b.ptr)
      return a
    end

    def add_constant(a : Vector({{ type_ }}), x : Number)
      LibGsl.{{ prefix }}_add_constant(a.ptr, x)
      return a
    end

    def sub(a : Vector({{ type_ }}), b : Vector)
      LibGsl.{{ prefix }}_sub(a.ptr, b.ptr)
      return a
    end

    def sub_constant(a : Vector({{ type_ }}), x : Number)
      LibGsl.{{ prefix }}_add_constant(a.ptr, -x)
      return a
    end

    def mul(a : Vector({{ type_ }}), b : Vector)
      LibGsl.{{ prefix }}_mul(a.ptr, b.ptr)
      return a
    end

    def mul_constant(a : Vector({{ type_ }}), x : Number)
      LibGsl.{{ prefix }}_scale(a.ptr, x)
      return a
    end

    def div(a : Vector({{ type_ }}), b : Vector)
      LibGsl.{{ prefix }}_div(a.ptr, b.ptr)
      return a
    end

    def div_constant(a : Vector({{ type_ }}), x : Number)
      LibGsl.{{ prefix }}_scale(a.ptr, 1 / x)
      return a
    end

  end
end

arithmetic_abstract LibGsl::GslVector, gsl_vector
arithmetic_abstract LibGsl::GslVectorInt, gsl_vector_int
