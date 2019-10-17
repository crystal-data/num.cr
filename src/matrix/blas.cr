require "../core/matrix/*"
require "./*"
require "../core/vector/*"

class Matrix(T, D)
  def mul(other : Vector)
    Bottle::Core::MatrixBlas.mul_vector(self, other)
  end

  def matmul(other : Matrix(T, D))
    Bottle::Core::MatrixBlas.mul_matrix(self, other)
  end
end
