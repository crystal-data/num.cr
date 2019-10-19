require "./*"
require "../gsl/*"
require "../blas/*"
require "../libs/gsl"

class Vector(T, D)
  def givens(other : Vector(T, D), da, db, c, s)
    LL.givens(self.copy, other.copy, da, db, c, s)
  end

  # Computes the dot product of two vectors
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [4.0, 5.0, 6.0]
  # v1.dot(v2) # => 32
  # ```
  def dot(other : Vector(T, D))
    LL.dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Vector.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    LL.norm(self)
  end

  # Sum of absolute values
  #
  # ```
  # v1 = Vector.new [-1, 1, 2]
  # v2.asum # => 4
  # ```
  def asum
    LL.asum(self)
  end

  # Index of absolute value max
  #
  # ```
  # v1 = Vector.new [-8, 1, 2]
  # v2.amax # => 0
  # ```
  def amax
    LL.amax(self)
  end
end
