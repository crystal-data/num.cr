require "../../libs/gsl"
require "../../vector/*"

module Bottle::Core::VectorIndex
  include Bottle::Core::Exceptions
  extend self

  # Gets a single element from a vector at a given index, the core
  # indexing operation of a vector
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[0] # => 1
  # ```
  def get_vector_element_at_index(vector : Pointer(LibGsl::GslVector), index : Int32)
    LibGsl.gsl_vector_get(vector, index)
  end

  # Sets a single element from a vector at a given index
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[0] = 10
  # vec # => [10, 2, 3]
  # ```
  def set_vector_element_at_index(vector : Pointer(LibGsl::GslVector), index : Int32, value : Number)
    LibGsl.gsl_vector_set(vector, index, value)
  end

  # Gets multiple elements from a vector at given indexes.  This returns
  # a `copy` since there is no way to create a contiguous slice of memory
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[1, 2]] # => [2, 3]
  # ```
  def get_vector_elements_at_indexes(vector : Pointer(LibGsl::GslVector), indexes : Array(Int32))
    sz = indexes.size
    ptv = LibGsl.gsl_vector_alloc(sz)
    indexes.each_with_index do |el, index|
      val = get_vector_element_at_index(vector, el)
      set_vector_element_at_index(ptv, index, val)
    end
    return Vector.new ptv, ptv.value.data
  end

  # Sets multiple elements of a vector by the given indexes.
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[0, 1]] = [10, 9]
  # vec # => [10, 9, 3]
  # ```
  def set_vector_element_at_indexes(vector : Pointer(LibGsl::GslVector), indexes : Array(Int32), values : Array(Number))
    indexes.each_with_index do |idx, index|
      LibGsl.gsl_vector_set(vector, idx, values[index])
    end
  end

  # Normalizes a slice range to allow indexing to work with gsl.  This means
  # bounding nil ranges and disallowing inclusive ranges
  def convert_range_to_slice(range : Range(Int32 | Nil, Int32 | Nil), size : UInt64)
    if !range.excludes_end?
      raise RangeError.new("Vectors do not support indexing with inclusive ranges. Always use '...'")
    end

    i = range.begin
    j = range.end

    start = (i.nil? ? 0 : i).as(UInt64 | Int32).to_u64
    finish = (j.nil? ? size : j).as(UInt64 | Int32).to_u64

    return start...finish
  end

  # Returns a view of a vector defined by a given range.  Currently only
  # supports single strided ranges due to limitations of Crystal
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[2...4] # => [3, 4]
  # ```
  def get_vector_elements_at_range(vector : Pointer(LibGsl::GslVector), range, size)
    newr = convert_range_to_slice(range, size)
    vv = LibGsl.gsl_vector_subvector(vector, newr.begin, newr.end - newr.begin)
    return Vector.new vv.vector, vv.vector.data
  end

  # Sets elements of a vector to given values based on the given range
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[1...] = [10, 9, 8, 7]
  # vec # => [1, 10, 9, 8, 7]
  # ```
  def set_vector_elements_at_range(vector : Pointer(LibGsl::GslVector), range, size, values)
    newr = convert_range_to_slice(range, size)
    newr.each_with_index do |el, index|
      LibGsl.gsl_vector_set(vector, el, values[index])
    end
  end

  # Returns a copy of a vector that owns its own memory
  #
  # ```
  # a = Vector.new [1, 2, 3]
  # a.copy # => [1, 2, 3]
  # ```
  def copy_vector(vector : Vector(LibGsl::GslVector, Float64))
    ptv = LibGsl.gsl_vector_alloc(vector.size)
    LibGsl.gsl_vector_memcpy(ptv, vector.ptr)
    return Vector.new ptv, ptv.value.data
  end
end
