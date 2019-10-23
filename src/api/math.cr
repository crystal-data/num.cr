require "../core/vector"
require "../blas/level_one"

module Bottle::B
  extend self

  # Elementwise addition of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # f1 + f2 # => Vector[3.0, 6.0, 9.0]
  # ```
  def add(a : Vector(U), b : Vector, inplace = false) forall U
    a = a.clone unless inplace
    {% if U == Float32 || U == Float64 %}
      b = b.astype(U)
      return BLAS.axpy(b, a)
    {% else %}
      return Vector.new(a.size) { |i| a[i] + b[i] }
    {% end %}
  end

  # Elementwise addition of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def add(a : Vector(U), x : Number, inplace = false) forall U
    a = a.clone unless inplace
    Vector.new(a.size) { |i| a[i] + x }
  end

  # Elementwise subtraction of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def subtract(a : Vector(U), b : Vector, inplace = false) forall U
    a = a.clone unless inplace
    {% if U == Float32 || U == Float64 %}
      b = b.astype(U)
      return BLAS.axpy(b, a, U.new(-1))
    {% else %}
      return Vector.new(a.size) { |i| a[i] - b[i] }
    {% end %}
  end

  # Elementwise subtraction of a Vector with a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def subtract(a : Vector(U), x : Number, inplace = false) forall U
    a = a.clone unless inplace
    Vector.new(a.size) { |i| a[i] - x }
  end

  # Elementwise multiplication of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def multiply(a : Vector(U), b : Vector, inplace = false) forall U
    a = a.clone unless inplace
    Vector.new(a.size) { |i| a[i] * b[i] }
  end

  # Elementwise multiplication of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def multiply(a : Vector(U), x : Number, inplace = false) forall U
    a = a.clone unless inplace
    {% if U == Float32 || U == Float64 %}
      return BLAS.scale(a, U.new(x))
    {% else %}
      return Vector.new(a.size) { |i| a[i] * x }
    {% end %}
  end

  # Elementwise division of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def divide(a : Vector(U), b : Vector, inplace = false) forall U
    a = a.clone unless inplace
    Vector.new(a.size) { |i| a[i] / b[i] }
  end

  # Elementwise division of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def divide(a : Vector(U), x : Number, inplace = false) forall U
    a = a.clone unless inplace
    {% if U == Float32 || U == Float64 %}
      return BLAS.scale(a, U.new(1/x))
    {% else %}
      return Vector.new(a.size) { |i| a[i] / x }
    {% end %}
  end

  def sqrt(a : Vector(U)) forall U
    Vector.new(a.size) { |i| Math.sqrt(a[i]) }
  end

  def power(a : Vector(U), power : Int32) forall U
    Vector.new(a.size) { |i| a[i]**power }
  end
end
