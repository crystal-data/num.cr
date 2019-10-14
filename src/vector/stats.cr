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
    Bottle::Core::VectorStats.vector_dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Vector.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    Bottle::Core::VectorStats.vector_norm(self)
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    Bottle::Core::VectorStats.vector_max(@ptr)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    Bottle::Core::VectorStats.vector_min(@ptr)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    Bottle::Core::VectorStats.vector_ptpv(@ptr)
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    Bottle::Core::VectorStats.vector_ptp(@ptr)
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmax # => 3
  # ```
  def idxmax
    Bottle::Core::VectorStats.vector_idxmax(@ptr)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmin # => 0
  # ```
  def idxmin
    Bottle::Core::VectorStats.vector_idxmin(@ptr)
  end

  # Computes the indexes of the minimum and maximum values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpidx # => {0, 3}
  # ```
  def ptpidx
    Bottle::Core::VectorStats.vector_ptpidx(@ptr)
  end

  def to_s(io)
    vals = (0...@size).map { |i| Bottle::Core::VectorIndex.get_vector_element_at_index(@ptr, i) }
    io << "[" << vals.join(", ") << "]"
  end
end
