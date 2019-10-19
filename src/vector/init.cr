require "./*"
require "../gsl/*"
require "../libs/gsl"

class Vector(T, D)
  getter data : Pointer(D)
  getter dtype : D.class
  getter obj : T
  getter owner : Int32
  getter ptr : Pointer(T)
  getter size : UInt64
  getter stride : UInt64

  # Converts a vector into a string representation.  This currently
  # displays all the values of a vector, but should re-worked to
  # truncate large vectors.
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # puts vec # => [1.0, 2.0, 3.0, 4.0, 5.0]
  # ```
  def to_s(io)
    vals = (0...@size).map { |i| LL.get(ptr, i) }
    io << "[" << vals.map { |v| v.round(3) }.join(", ") << "]"
  end

  # Returns a copy of a vector that owns its own memory
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec.copy # => [1.0, 2.0, 3.0, 4.0, 5.0]
  # ```
  def copy
    v = LL.allocate(D, T, size)
    LL.memcpy(v, ptr)
    return Vector.new v, v.value.data
  end

  # Reverses a copy of the vector and returns the copy.  This
  # could return an in-place slice and work with BLAS methods,
  # but GSL does not support negative strides on vectors so
  # this is currently a copy.
  #
  # ```
  # vec = Vector.new [1, 2, 3, 4, 5]
  # vec.reverse [5.0, 4.0, 3.0, 2.0, 1.0]
  # ```
  def reverse
    res = self.copy
    LL.reverse(res.ptr)
    return res
  end

  # Handles different dtype indexable inputs and returns the valid
  # array type.  If an array has a heterogenous type, Float64 is
  # the "catch all" dtype
  def self.new(data : Indexable(A)) forall A
    new(*infer_type(data))
  end

  # Allocates and sets the values of a vector of Doubles
  def self.infer_type(data : Indexable(Float64))
    v = LL.allocate(Float64, LibGsl::GslVector, data.size)
    data.each_with_index { |e, i|  LL.set(v, i, e) }
    return v, v.value.data
  end

  # Allocates and sets the values of a vector of Reals
  def self.infer_type(data : Indexable(Float32))
    v = LL.allocate(Float32, LibGsl::GslVectorFloat, data.size)
    data.each_with_index { |e, i|  LL.set(v, i, e) }
    return v, v.value.data
  end

  # Alocates and sets the values of a vector of ComplexDoubles
  def self.infer_type(data : Indexable(LibGsl::ComplexDouble))
    v = LL.allocate(Complex, LibGsl::GslVectorComplex, data.size)
    data.each_with_index { |e, i|  LL.set(v, i, e) }
    return v, v.value.data
  end

  # The "catch all" constructor.  Allocates a vector of Doubles
  # from heterogenous inputs.
  def self.infer_type(data : Indexable(Number))
    v = LL.allocate(Float64, LibGsl::GslVector, data.size)
    data.each_with_index { |e, i|  LL.set(v, i, e) }
    return v, v.value.data
  end

  def finalize
    LL.free(ptr)
  end

  # One type of valid initialization takes a pointer to a
  # GslVector and a pointer to its data.  This is the most
  # common init methods, the other is used primarily if
  # taking a view of an existing vector, and is never called
  # from a user facing method.
  def initialize(@ptr : Pointer(T), @data : Pointer(D))
    @obj = @ptr.value
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
    @dtype = D
  end

  # Initializes a vector from a GslVector and a pointer to its
  # data.  This is primary used when initializing a vector from
  # a vector view, which provides direct access to a vector, not
  # through a pointer.
  def initialize(@obj : T, @data : Pointer(D))
    @ptr = pointerof(@obj)
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
    @dtype = D
  end

  # Yields each index of a vector, useful for providing a base
  # method to other iterators through a vector
  def each_index(*, all = false, &block)
    size.times do |i|
      yield i
    end
  end

  # Lazily yields each element of a vector
  def each(*, all = false, &block)
    each_index(all: all) { |i| yield LL.get(ptr, i) }
  end

  # Lazily yields each element of a vector, along with its
  # index.
  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i| yield LL.get(ptr, i) }
  end

  # Initializes a vector of size `n` containing all zeros.
  # This currently only creates a vector of doubles.
  #
  # ```
  # vec = Vector.zeros(5)
  # vec # => [0.0, 0.0, 0.0, 0.0, 0.0]
  # ```
  def self.zeros(n : Indexer)
    v = LL.zero(Float64, LibGsl::GslVector, n)
    return Vector.new v, v.value.data
  end

  # Initializes a vector of size `n` where each entry is
  # set to the provided `fill_value`.  Currently can only
  # create a vector of doubles.
  #
  # ```
  # vec = Vector.full(2, 3)
  # vec # => [3.0, 3.0]
  # ```
  def self.full(n : Indexer, fill_value : BNum)
    v = LL.allocate(Float64, LibGsl::GslVector, n)
    LL.full(v, fill_value)
    return Vector.new v, v.value.data
  end

  # Initializes a basis vector of size `n` with the `ith` element
  # set to 1.0
  #
  # ```
  # vec = Vector.basis(5, 0)
  # vec # => [1.0, 0.0, 0.0, 0.0, 0.0]
  # ```
  def self.basis(n : Indexer, i : Indexer)
    v = LL.allocate(Float64, LibGsl::GslVector, n)
    LL.set_basis(v, i)
    return Vector.new v, v.value.data
  end
end
