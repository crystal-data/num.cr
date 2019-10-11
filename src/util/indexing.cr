require "../llib/gsl"
require "../core/vector"
require "./exceptions"

# Creates indexing methods for each support data type of gsl.
# Since all the calls are virtually identical, and since
# only the pointers are passed around, this is an easy way
# to avoid duplicate code for no reason.
macro indexing_abstract(type_, prefix)
  module Bottle::Util::Indexing
    include Bottle::Util::Exceptions
    extend self

    # Gets a single element from a vector at a given index, the core
    # indexing operation of a vector
    #
    # ```
    # vec = Vector.new [1, 2, 3, 4, 5]
    # vec[0] # => 1
    # ```
    def get_vector_element_at_index(vector : Pointer({{ type_ }}), index : Int32)
      LibGsl.{{ prefix }}_get(vector, index)
    end

    # Sets a single element from a vector at a given index
    #
    # ```
    # vec = Vector.new [1, 2, 3]
    # vec[0] = 10
    # vec # => [10, 2, 3]
    # ```
    def set_vector_element_at_index(vector : Pointer({{ type_ }}), index : Int32, value : Number)
      LibGsl.{{ prefix }}_set(vector, index, value)
    end

    # Gets multiple elements from a vector at given indexes.  This returns
    # a `copy` since there is no way to create a contiguous slice of memory
    #
    # ```
    # vec = Vector.new [1, 2, 3]
    # vec[[1, 2]] # => [2, 3]
    # ```
    def get_vector_elements_at_indexes(vector : Pointer({{ type_ }}), indexes : Array(Int32))
      sz = indexes.size
      ptv = LibGsl.{{ prefix }}_alloc(sz)
      indexes.each_with_index do |el, index|
        val = get_vector_element_at_index(vector, el)
        set_vector_element_at_index(ptv, index, val)
      end
      return Vector.new ptv
    end

    # Sets multiple elements of a vector by the given indexes.
    #
    # ```
    # vec = Vector.new [1, 2, 3]
    # vec[[0, 1]] = [10, 9]
    # vec # => [10, 9, 3]
    # ```
    def set_vector_element_at_indexes(vector : Pointer({{ type_ }}), indexes : Array(Int32), values : Array(Number))
      indexes.each_with_index do |idx, index|
        LibGsl.{{ prefix }}_set(vector, idx, values[index])
      end
    end

    # Normalizes a slice range to allow indexing to work with gsl.  This means
    # bounding nil ranges and disallowing inclusive ranges
    def convert_range_to_slice(range : Range(Int32| Nil, Int32 | Nil), size : UInt64)
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
    def get_vector_elements_at_range(vector : Pointer({{ type_ }}), range, size)
      newr = convert_range_to_slice(range, size)
      vv = LibGsl.{{ prefix }}_subvector(vector, newr.begin, newr.end - newr.begin)
      return Vector.new vv.vector
    end

    # Sets elements of a vector to given values based on the given range
    #
    # ```
    # vec = Vector.new [1, 2, 3, 4, 5]
    # vec[1...] = [10, 9, 8, 7]
    # vec # => [1, 10, 9, 8, 7]
    # ```
    def set_vector_elements_at_range(vector : Pointer({{ type_ }}), range, size, values)
      newr = convert_range_to_slice(range, size)
      newr.each_with_index do |el, index|
        LibGsl.{{ prefix }}_set(vector, el, values[index])
      end
    end

    def copy_vector(vector : Vector({{ type_ }}))
      ptv = LibGsl.{{ prefix }}_alloc(vector.size)
      LibGsl.{{ prefix }}_memcpy(ptv, vector.ptr)
      return Vector.new(ptv.value)
    end

  end
end

indexing_abstract LibGsl::GslVector, gsl_vector
indexing_abstract LibGsl::GslVectorInt, gsl_vector_int
