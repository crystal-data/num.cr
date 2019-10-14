require "../core/vector/*"
require "./*"

class Vector(T, D)
  # Computes the dot product of two vectors
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [4.0, 5.0, 6.0]
  # v1.dot(v2) # => 32
  # ```
  def dot(other : Vector)
    Bottle::Core::VectorBlas.vector_dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Vector.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    Bottle::Core::VectorBlas.vector_norm(self)
  end

  # Sum of absolute values
  #
  # ```
  # v1 = Vector.new [-1, 1, 2]
  # v2.asum # => 4
  # ```
  def asum
    Bottle::Core::VectorBlas.vector_absolute_sum(self)
  end

  # Index of absolute value max
  #
  # ```
  # v1 = Vector.new [-8, 1, 2]
  # v2.amax # => 0
  # ```
  def amax
    Bottle::Core::VectorBlas.vector_absolute_max(self)
  end
end
