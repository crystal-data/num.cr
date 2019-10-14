require "../libs/gsl"
require "../core/vector/*"
require "./*"

class Vector(T, D)
  def copy
    Bottle::Core::VectorIndex.copy_vector(self)
  end

  # Gets a single element from a vector at a given index, the core
  # indexing operation of a vector
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[0] # => 1
  # ```
  def [](index : Int32)
    Bottle::Core::VectorIndex.get_vector_element_at_index(@ptr, index)
  end

  # Gets multiple elements from a vector at given indexes.  This returns
  # a `copy` since there is no way to create a contiguous slice of memory
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[1, 2]] # => [2, 3]
  # ```
  def [](indexes : Array(Int32))
    Bottle::Core::VectorIndex.get_vector_elements_at_indexes(@ptr, indexes)
  end

  # Returns a view of a vector defined by a given range.  Currently only
  # supports single strided ranges due to limitations of Crystal
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[2...4] # => [3, 4]
  # ```
  def [](range : Range(Int32 | Nil, Int32 | Nil))
    Bottle::Core::VectorIndex.get_vector_elements_at_range(@ptr, range, @size)
  end

  # Sets a single element from a vector at a given index
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[0] = 10
  # vec # => [10, 2, 3]
  # ```
  def []=(index : Int32, value : Number)
    Bottle::Core::VectorIndex.set_vector_element_at_index(@ptr, index, value)
  end

  # Sets multiple elements of a vector by the given indexes.
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[0, 1]] = [10, 9]
  # vec # => [10, 9, 3]
  # ```
  def []=(indexes : Array(Int32), values : Array(Number))
    Bottle::Core::VectorIndex.set_vector_elements_at_indexes(@ptr, indexes, values)
  end

  # Sets elements of a vector to given values based on the given range
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[1...] = [10, 9, 8, 7]
  # vec # => [1, 10, 9, 8, 7]
  # ```
  def []=(range : Range(Int32 | Nil, Int32 | Nil), values : Array(Number))
    Bottle::Core::VectorIndex.set_vector_elements_at_range(@ptr, range, @size, values)
  end
end
