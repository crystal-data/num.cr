require "../core/vector"
require "../blas/level_one"

module Bottle::B
  extend self

  # Elementwise addition of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # add(f1, f2) # => Vector[3.0, 6.0, 9.0]
  # ```
  def add(a : Vector(U), b : Vector) forall U
    {% if U == Float32 || U == Float64 %}
      a = a.clone
      b = b.astype(U)
      return BLAS.axpy(b, a)
    {% else %}
      return Vector.new(a.size) { |i| a[i] + b[i] }
    {% end %}
  end

  # Elementwise addition of a matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # add(j1, j1) # => [[2, 4], [6, 8]]
  # ```
  def add(a : Matrix(U), b : Matrix) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] + b[i, j] }
  end

  # Elementwise addition of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # add(f1, 2) # => [3.0, 4.0, 5.0]
  # ```
  def add(a : Vector(U), x : Number) forall U
    Vector.new(a.size) { |i| a[i] + x }
  end

  # Elementwise addition of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # add(j1, 2) # => [[3, 4], [5, 6]]
  # ```
  def add(a : Matrix(U), x : Number) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] + x }
  end

  # Elementwise subtraction of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # subtract(f1, f2) # => [-1.0, -2.0, -3.0]
  # ```
  def subtract(a : Vector(U), b : Vector) forall U
    {% if U == Float32 || U == Float64 %}
      a = a.clone
      b = b.astype(U)
      return BLAS.axpy(b, a, U.new(-1))
    {% else %}
      return Vector.new(a.size) { |i| a[i] - b[i] }
    {% end %}
  end

  # Elementwise subtraction of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # subtract(j1, j1) # => [[0, 0], [0, 0]]
  # ```
  def subtract(a : Matrix(U), b : Matrix) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] - b[i, j] }
  end

  # Elementwise subtraction of a Vector with a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # subtract(f2, 2) # => [-1.0, 0.0, 1.0]
  # ```
  def subtract(a : Vector(U), x : Number) forall U
    Vector.new(a.size) { |i| a[i] - x }
  end

  # Elementwise subtraction of a Matrix with a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # subtract(j1, 2) # => [[-1, 0], [1, 2]]
  # ```
  def subtract(a : Matrix(U), x : Number) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] - x }
  end

  # Elementwise multiplication of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # multiply(f1, f2) # => [3.0, 8.0, 18.0]
  # ```
  def multiply(a : Vector(U), b : Vector) forall U
    Vector.new(a.size) { |i| a[i] * b[i] }
  end

  # Elementwise multiplication of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # multiply(j1, j1) # => [[1, 4], [9, 16]]
  # ```
  def multiply(a : Matrix(U), b : Matrix) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] * b[i, j] }
  end

  # Elementwise multiplication of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # multiply(f1, 2) # => [2.0, 4.0, 6.0]
  # ```
  def multiply(a : Vector(U), x : Number) forall U
    {% if U == Float32 || U == Float64 %}
      a = a.clone
      return BLAS.scale(a, U.new(x))
    {% else %}
      return Vector.new(a.size) { |i| a[i] * x }
    {% end %}
  end

  # Elementwise multiplication of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # multiply(j1, 2) # => [[2, 4], [6, 8]]
  # ```
  def multiply(a : Matrix(U), x : Number) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] * x }
  end

  # Elementwise division of a Vector to another equally sized Vector
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # f2 = Vector.new [2.0, 4.0, 6.0]
  # divide(f1, f2) # => [0.5, 0.5, 0.5]
  # ```
  def divide(a : Vector(U), b : Vector) forall U
    Vector.new(a.size) { |i| a[i] / b[i] }
  end

  # Elementwise division of a Matrix to another equally sized Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # divide(j1, j2) # => [[1, 1], [1, 1]]
  # ```
  def divide(a : Matrix(U), b : Matrix) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] / b[i, j] }
  end

  # Elementwise division of a Vector to a scalar
  #
  # ```
  # f1 = Vector.new [1.0, 2.0, 3.0]
  # divide(f1, 2) # => [0.5, 1, 1.5]
  # ```
  def divide(a : Vector(U), x : Number) forall U
    {% if U == Float32 || U == Float64 %}
      a = a.clone
      return BLAS.scale(a, U.new(1/x))
    {% else %}
      return Vector.new(a.size) { |i| a[i] / x }
    {% end %}
  end

  # Elementwise division of a Matrix to a scalar
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # divide(j1, 2) # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def divide(a : Matrix(U), x : Number) forall U
    return Matrix.new(a.nrows, a.ncols) { |i, j| a[i, j] / x }
  end

  # Elementwise square root of a Vector
  #
  # ```
  # f1 = Vector.new [1.0, 4.0, 9.0]
  # sqrt(f1) # => [1.0, 2.0, 3.0]
  # ```
  def sqrt(a : Vector(U)) forall U
    Vector.new(a.size) { |i| Math.sqrt(a[i]) }
  end

  # Elementwise square root of a Matrix
  #
  # ```
  # j1 = Matrix.new [[1, 4], [16, 25]]
  # sqrt(f1) # => [[1, 2], [4, 5]]
  # ```
  def sqrt(a : Matrix(U)) forall U
    Matrix.new(a.ncols, a.nrows) { |i, j| Math.sqrt(a[i, j]) }
  end

  # Raise a vector to a power
  #
  # ```
  # f1 = Vector.new [1.0, 4.0, 9.0]
  # power(f1, 2) # => [1.0, 16.0, 81.0]
  # ```
  def power(a : Vector(U), x : Number) forall U
    Vector.new(a.size) { |i| a[i]**x }
  end

  # Raise a matrix to a power
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # power(j1, 2) # => [[1, 4], [9, 16]]
  # ```
  def power(a : Matrix(U), x : Number) forall U
    Matrix.new(a.ncols, a.nrows) { |i, j| a[i, j]**x }
  end

  # Raise a scalar to the power of a vector
  #
  # ```
  # f1 = Vector.new [1.0, 4.0, 9.0]
  # power(2, f1) # => [2.0, 16.0, 512.0]
  # ```
  def power(x : Number, a : Vector(U)) forall U
    Vector.new(a.size) { |i| x**a[i] }
  end

  # Raise a scalar to the power of a matrix
  #
  # ```
  # j1 = Matrix.new [[1, 2], [3, 4]]
  # power(2, j1) # => [[2, 4], [8, 16]]
  # ```
  def power(a : Matrix(U), x : Number) forall U
    Matrix.new(a.ncols, a.nrows) { |i, j| x**a[i, j] }
  end
end
