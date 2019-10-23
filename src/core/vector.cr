require "./object"

module Bottle
  SUPPORTED_TYPES = {Float32, Float64, Complex}
end

class Bottle::Vector(T) < Bottle::Internal::BottleObject(T)
  # Crystal slice pointing to the start of the vector’s data.
  getter data : Slice(T)

  # Number of elements in a Vector
  getter size : Int32

  # Flag indicating if a Vector is a view of another `Matrix` or `Vector`
  getter owner : Bool

  # Offset between consecutive values in a Vector
  getter stride : Int32

  # Creates a new `Vector` of an arbitrary size from a given
  # indexable *data*.  The type of the Vector is inferred from
  # the provided data, as are the size, and stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v         # => Vector[1, 2, 3, 4, 5]
  # typeof(v) # => Vector(Int32)
  # ```
  def initialize(data : Indexable(T))
    check_type
    @size = data.size
    @data = Slice(T).new(size) { |i| data[i] }
    @stride = 1
    @owner = true
  end

  # Creates a new `Vector` from a slice and strides.  This is
  # a lower level method that won't be used often by users,
  # but is very handy in cloning vectors, and creating vectors
  # from low level C libraries.
  #
  # ```
  # s = Slice.new(5) { |i| i }
  # v = Vector.new s, 1, true
  #
  # v         # => Vector[1, 2, 3, 4, 5]
  # typeof(v) # => Vector(Int32)
  # ```
  def initialize(@data : Slice(T), @stride = 1, @owner = true)
    check_type
    @size = @data.size // @stride
  end

  # Creates a new `Vector` from a block.  Infers the type
  # of the Vector from the value yielded by the block.
  #
  # ```
  # v = Vector.new(5) { |i| i / 2 }
  #
  # v         # => Vector[0.0, 0.5, 1.0, 1.5, 2.0]
  # typeof(v) # => Vector(Float64)
  # ```
  def self.new(size : Int32, &block : Int32 -> T)
    data = Slice(T).new(size) { |i| yield i }
    new(data, 1, true)
  end

  # Selects a single value from a vector. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # get(3) # => 4
  # ```
  private def get(i)
    check_sign(i)
    @data[i * stride]
  end

  # Sets a single value of a vector. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # set(0, 8)
  # v # => Vector[8, 2, 3, 4, 5]
  # ```
  private def set(i, x)
    check_sign(i)
    @data[i * stride] = T.new(x)
  end

  # Selects a slice from a vector. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # get_n(0, 3) # => Slice[1, 2, 3]
  # ```
  private def get_n(i, n)
    check_sign(i)
    @data[i * stride, n * stride]
  end

  # Sets multiple values of a vector. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # set_n([0, 1, 2], [10, 10, 10])
  # v # => Vector[10, 10, 10, 4, 5]
  # ```
  private def set_n(is, xs)
    is.each_with_index { |e, i| set(e, xs[i]) }
  end

  # Selects a slice from a vector. Calculates the offset
  # of the elements from a given range, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # get_slice(1...) # => Slice[2, 3, 4, 5]
  # ```
  private def get_slice(slice)
    i, n = range_to_slice(slice, size)
    get_n(i, n - i)
  end

  # Sets multiple values of a vector. Calculates the offset
  # of the elements from a given slice, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # set_slice(...3, [10, 10, 10])
  # v # => Vector[10, 10, 10, 4, 5]
  # ```
  private def set_slice(slice, xs)
    i, n = range_to_slice(slice, size)
    (i...n).each_with_index { |e, i| set(e, xs[i]) }
  end

  # Selects a single value from a vector. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[3] # => 4
  # ```
  def [](i : Int32)
    get(i)
  end

  # Sets a single value of a vector. Calculates the offset
  # of the element from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[0] = 8
  # v # => Vector[8, 2, 3, 4, 5]
  # ```
  def []=(i : Int32, x : Number)
    set(i, x)
  end

  # Selects multiple non-contiguous elements from a
  # Vector.  This method returns a copy since the memory
  # is not aligned.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[[0, 1, 2]] # => Vector[1, 2, 3]
  # v.owner      # => true
  # ```
  def [](is)
    Vector.new(is.size) { |i| self[is[i]] }
  end

  # Sets multiple values of a vector. Calculates the offset
  # of the elements from a given index, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[[0, 1, 2]] = [10, 10, 10]
  # v # => Vector[10, 10, 10, 4, 5]
  # ```
  def []=(is, xs)
    set_n(is, xs)
  end

  # Selects a slice from a vector. Calculates the offset
  # of the elements from a given range, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[1...] # => Vector[2, 3, 4, 5]
  # ```
  def [](is : Range(Int32?, Int32?))
    Vector.new(get_slice(is), stride, false)
  end

  # Sets multiple values of a vector. Calculates the offset
  # of the elements from a given slice, and the stride.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4, 5]
  #
  # v[...3] = [10, 10, 10]
  # v # => Vector[10, 10, 10, 4, 5]
  # ```
  def []=(is : Range(Int32?, Int32?), xs)
    set_slice(is, xs)
  end

  # Computes the string representation of a Vector.  This currently
  # displays all the values of a Vector, but should re-worked to
  # truncate large Vectors
  #
  # ```
  # f = Vector.new [1, 2, 3, 4, 5]
  # puts f # => Vector[1, 2, 3, 4, 5]
  # ```
  def to_s(io)
    io << "Vector[" << (0...size).map { |i| self[i] }.join(", ") << "]"
  end

  # Lazily yields the index values of a Vector.  This method
  # is used as the core iteration method for Vectors
  #
  # ```crystal
  # v = Vector.new [1, 2, 3, 4, 5]
  # v.each_index { |i| puts i }
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # # 4
  # ```
  def each_index(*, all = false, &block)
    size.times do |i|
      yield i
    end
  end

  # Lazily yields a Vector and its index, one element at a time.
  # The core iteration method for a Vector
  #
  # ```crystal
  # v = Vector.new [1, 2, 3, 4, 5]
  # v.each { |e| puts e }
  #
  # # 1
  # # 2
  # # 3
  # # 4
  # # 5
  # ```
  def each(*, all = false, &block)
    each_index(all: all) { |i| yield self[i] }
  end

  # Lazily yields a Vector and its index, one element at a time.
  # Useful for reduction methods that require both elements to
  # do work on, and the indices they belong to.
  #
  # ```crystal
  # v = Vector.new [1, 2, 3, 4, 5]
  # v.each_with_index { |e, i| puts "#{e}_#{i}" }
  #
  # # 1_0
  # # 2_1
  # # 3_2
  # # 4_3
  # # 5_4
  # ```
  def each_with_index(*, all = false, &block)
    each_index(all: all) { |i| yield(self[i], i) }
  end

  # Initializes a Vector with an uninitialized slice
  # of data.  This is just another alias for zeros,
  # since Crystal right now doesn't support slices
  # pointing to generic memory, but if Crystal does
  # support this down the road, this will change
  #
  # ```crystal
  # f = Vector.empty(5, dtype: Int32)
  # f # => Vector[0, 0, 0, 0, 0]
  # ```
  def self.empty(n : Int32, dtype : U.class = Float64) forall U
    Vector.new Slice(U).new(n), 1, true
  end

  # Initializes a vector full of zeros.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # f = Vector.zeros(5, dtype: Int32)
  # f # => Vector[0, 0, 0, 0, 0]
  # ```
  def self.zeros(n : Int32, dtype : U.class = Float64) forall U
    Vector.new Slice(U).new(n), 1, true
  end

  # Initializes a vector full of ones.  Default
  # dtype is Float64, but other dtypes are supported
  #
  # ```crystal
  # f = Flask.ones(5, dtype: Int32)
  # f # => Vector[1, 1, 1, 1, 1]
  # ```
  def self.ones(n : Indexer, dtype : U.class = Float64) forall U
    Vector.new Slice(U).new(n, U.new(1)), 1, true
  end

  # Initializes a vector full of a given scalar.  Default
  # dtype is Float64, but other dtypes are supported
  #
  # ```crystal
  # f = Vector.full(5, 5, dtype: Int32)
  # f # => Vector[5, 5, 5, 5, 5]
  # ```
  def self.full(n : Int32, x : Number, dtype : U.class = Float64) forall U
    Vector.new Slice(U).new(n, U.new(x)), 1, true
  end

  # Pours a flask full of random data.
  # The dtype of the flask is inferred
  # from the values on either end of the
  # range.
  #
  # ```crystal
  # f = Vector.random(0, 10, 5)
  # f # => Vector[4, 8, 7, 8, 4]
  # ```
  def self.random(r : Range(U, U), n : Int32) forall U
    Vector.new(n) { |_| Random.rand(r) }
  end

  # Returns a copy of a Vector that owns its own memory
  #
  # ```
  # f = Vector.new [1, 2, 3, 4, 5]
  # f.clone # => Vector[1, 2, 3, 4, 5]
  # ```
  def clone
    Vector.new(size) { |i| self[i] }
  end

  # Casts a Vector to another data dtype.
  # If the Vector is already the given dtype,
  # a copy is not made, otherwise a new Vector
  # is returned.
  #
  # ```crystal
  # f = Vector.new [1, 2, 3]
  # f.astype(Float64) # => Vector[1.0, 2.0, 3.0]
  # ```
  def astype(dtype : U.class) forall U
    if T == U
      return self
    end
    Vector.new(self.size) { |i| U.new(self[i]) }
  end

  # Reverses a copy of the Vector and returns the copy.
  #
  # ```
  # f = Vector.new [1, 2, 3, 4, 5]
  # f.reverse # => Vector[5, 4, 3, 2, 1]
  # ```
  def reverse
    f = clone
    f.data.reverse!
    f
  end
end
