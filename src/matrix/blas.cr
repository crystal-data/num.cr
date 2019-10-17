require "../core/matrix/*"
require "./*"
require "../core/vector/*"

class Matrix(T, D)
  # Multiplies a matrix times a vector
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # v = Vector.new [1, 2, 3]
  # m.mul(v) # => [14.0, 32.0, 50.0]
  # ```
  def mul(other : Vector)
    Bottle::Core::MatrixBlas.mul_vector(self, other)
  end

  # Multiplies a matrix times a matrix
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  # m.matmul(m) # => [[30.0, 36.0, 42.0], [66.0, 81.0, 96.0], [102.0, 126.0, 150.0]]
  # ```
  def matmul(other : Matrix(T, D))
    Bottle::Core::MatrixBlas.mul_matrix(self, other)
  end
end
