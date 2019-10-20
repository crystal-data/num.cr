require "../libs/dtype"
require "./*"
require "../blas/*"

class Flask(T)
  # Computes the dot product of two vectors
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [4.0, 5.0, 6.0]
  # f1.dot(f2) # => 32
  # ```
  def dot(other : Flask(T))
    LL.dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Flask.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    LL.norm(self)
  end

  # Sum of absolute values
  #
  # ```
  # f1 = Flask.new [-1, 1, 2]
  # f2.asum # => 4
  # ```
  def asum
    LL.asum(self)
  end

  # Index of absolute value max
  #
  # ```
  # f1 = Flask.new [-8, 1, 2]
  # f2.amax # => 0
  # ```
  def amax
    LL.amax(self)
  end
end
