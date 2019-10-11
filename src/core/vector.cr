require "../llib/gsl"
require "../util/indexing"
require "../util/arithmetic"
require "../util/statistics"
require "../util/generic"
require "../util/blas_arithmetic"

class Vector(T)
  @ptr : Pointer(T)
  @obj : T
  @owner : Int32
  @size : UInt64
  @stride : UInt64

  getter ptr
  getter owner
  getter size
  getter strides

  def self.new(data : Array(A)) forall A
    new(fetch_struct(data))
  end

  private def self.fetch_struct(items : Array(Int32))
    vector = LibGsl.gsl_vector_int_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_int_set(vector, i, e) }
    return vector
  end

  private def self.fetch_struct(items : Array(Float64))
    vector = LibGsl.gsl_vector_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
    return vector
  end

  private def self.fetch_struct(items : Array(Float64 | Int32))
    vector = LibGsl.gsl_vector_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
    return vector
  end

  def initialize(@ptr : Pointer(T))
    @obj = @ptr.value
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
  end

  def initialize(@obj : T)
    @ptr = pointerof(@obj)
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
  end

  def copy
    Bottle::Util::Indexing.copy_vector(self)
  end

  # Gets a single element from a vector at a given index, the core
  # indexing operation of a vector
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[0] # => 1
  # ```
  def [](index : Int32)
    Bottle::Util::Indexing.get_vector_element_at_index(@ptr, index)
  end

  # Gets multiple elements from a vector at given indexes.  This returns
  # a `copy` since there is no way to create a contiguous slice of memory
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[1, 2]] # => [2, 3]
  # ```
  def [](indexes : Array(Int32))
    Bottle::Util::Indexing.get_vector_elements_at_indexes(@ptr, indexes)
  end

  # Returns a view of a vector defined by a given range.  Currently only
  # supports single strided ranges due to limitations of Crystal
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[2...4] # => [3, 4]
  # ```
  def [](range : Range(Int32 | Nil, Int32 | Nil))
    Bottle::Util::Indexing.get_vector_elements_at_range(@ptr, range, @size)
  end

  # Sets a single element from a vector at a given index
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[0] = 10
  # vec # => [10, 2, 3]
  # ```
  def []=(index : Int32, value : Number)
    Bottle::Util::Indexing.set_vector_element_at_index(@ptr, index, value)
  end

  # Sets multiple elements of a vector by the given indexes.
  #
  # ```
  # vec = Vector.new [1, 2, 3]
  # vec[[0, 1]] = [10, 9]
  # vec # => [10, 9, 3]
  # ```
  def []=(indexes : Array(Int32), values : Array(Number))
    Bottle::Util::Indexing.set_vector_elements_at_indexes(@ptr, indexes, values)
  end

  # Sets elements of a vector to given values based on the given range
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec[1...] = [10, 9, 8, 7]
  # vec # => [1, 10, 9, 8, 7]
  # ```
  def []=(range : Range(Int32 | Nil, Int32 | Nil), values : Array(Number))
    Bottle::Util::Indexing.set_vector_elements_at_range(@ptr, range, @size, values)
  end

  # Elementwise addition of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 + v2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Vector(T))
    Bottle::Util::VectorMath.add(self.copy, other)
  end

  # Elementwise addition of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 + v2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : Number)
    Bottle::Util::VectorMath.add_constant(self.copy, other)
  end

  # Elementwise subtraction of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 - v2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Vector(T))
    Bottle::Util::VectorMath.sub(self.copy, other)
  end

  # Elementwise subtraction of a vector with a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 - v2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : Number)
    Bottle::Util::VectorMath.sub_constant(self.copy, other)
  end

  # Elementwise multiplication of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 * v2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Vector(T))
    Bottle::Util::VectorMath.mul(self.copy, other)
  end

  # Elementwise multiplication of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 + v2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : Number)
    Bottle::Util::VectorMath.mul_constant(self.copy, other)
  end

  # Elementwise division of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 / v2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Vector(T))
    Bottle::Util::VectorMath.div(self.copy, other)
  end

  # Elementwise division of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 / v2 # => [0.5, 1, 1.5]
  # ```
  def /(other : Number)
    Bottle::Util::VectorMath.div_constant(self.copy, other)
  end

  # Computes the dot product of two vectors
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [4.0, 5.0, 6.0]
  # v1.dot(v2) # => 32
  # ```
  def dot(other : Vector)
    Bottle::Util::VectorMath.vector_dot(self, other)
  end

  # Computes the euclidean norm of the vector
  #
  # ```
  # vec = Vector.new [1.0, 2.0, 3.0]
  # vec.norm # => 3.741657386773941
  # ```
  def norm
    Bottle::Util::VectorMath.vector_norm(self)
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    Bottle::Util::VectorStats.vector_max(@ptr)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    Bottle::Util::VectorStats.vector_min(@ptr)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    Bottle::Util::VectorStats.vector_ptpv(@ptr)
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    Bottle::Util::VectorStats.vector_ptp(@ptr)
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmax # => 3
  # ```
  def idxmax
    Bottle::Util::VectorStats.vector_idxmax(@ptr)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmin # => 0
  # ```
  def idxmin
    Bottle::Util::VectorStats.vector_idxmin(@ptr)
  end

  # Computes the indexes of the minimum and maximum values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpidx # => {0, 3}
  # ```
  def ptpidx
    Bottle::Util::VectorStats.vector_ptpidx(@ptr)
  end

  def to_s(io)
    vals = (0...@size).map { |i| Bottle::Util::Indexing.get_vector_element_at_index(@ptr, i) }
    io << "[" << vals.join(", ") << "]"
  end
end
