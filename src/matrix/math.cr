require "../libs/gsl"
require "../core/matrix/*"
require "./*"

class Matrix(T, D)
  # Elementwise addition of a matrix to another equally sized matrix
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = Matrix.new [[2.0, 4.0, 6.0]]
  # v1 + v2 # => [[3.0, 6.0, 9.0]]
  # ```
  def +(other : Matrix(T, D))
    Bottle::Core::MatrixMath.add(self.copy, other)
  end

  # Elementwise addition of a Matrix to a scalar
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = 2
  # v1 + v2 # => [[3.0, 4.0, 5.0]]
  # ```
  def +(other : Number)
    Bottle::Core::MatrixMath.add_constant(self.copy, other.to_f)
  end

  # Elementwise subtraction of a Matrix to another equally sized Matrix
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = Matrix.new [[2.0, 4.0, 6.0]]
  # v1 - v2 # => [[-1.0, -2.0, -3.0]]
  # ```
  def -(other : Matrix(T, D))
    Bottle::Core::MatrixMath.sub(self.copy, other)
  end

  # Elementwise subtraction of a Matrix with a scalar
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = 2
  # v1 - v2 # => [[-1.0, 0.0, 1.0]]
  # ```
  def -(other : Number)
    Bottle::Core::MatrixMath.sub_constant(self.copy, other.to_f)
  end

  # Elementwise multiplication of a Matrix to another equally sized Matrix
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = Matrix.new [[2.0, 4.0, 6.0]]
  # v1 * v2 # => [[3.0, 8.0, 18.0]]
  # ```
  def *(other : Matrix(T, D))
    Bottle::Core::MatrixMath.mul_elements(self.copy, other)
  end

  # Elementwise multiplication of a Matrix to a scalar
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = 2
  # v1 + v2 # => [[2.0, 4.0, 6.0]]
  # ```
  def *(other : Number)
    Bottle::Core::MatrixMath.mul_constant(self.copy, other)
  end

  # Elementwise division of a Matrix to another equally sized Matrix
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = Matrix.new [[2.0, 4.0, 6.0]]
  # v1 / v2 # => [[0.5, 0.5, 0.5]]
  # ```
  def /(other : Matrix(T, D))
    Bottle::Core::MatrixMath.div_elements(self.copy, other)
  end

  # Elementwise division of a Matrix to a scalar
  #
  # ```
  # v1 = Matrix.new [[1.0, 2.0, 3.0]]
  # v2 = 2
  # v1 / v2 # => [[0.5, 1, 1.5]]
  # ```
  def /(other : Number)
    Bottle::Core::MatrixMath.div_constant(self.copy, other.to_f)
  end
end
