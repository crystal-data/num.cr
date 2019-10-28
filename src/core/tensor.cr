require "../api/vectorprint"
require "./ufunc"

# A container that allows accessing elements via a numeric index.
#
# Indexing starts at 0. A negative index is assumed to be
# relative to the end of the container: -1 indicates the last element,
# -2 is the next to last element, and so on.
#
# Types including this module are typically `Array`-like types.
struct Bottle::Tensor(T)
  # Returns the number of elements in this object
  getter size : Int32

  @capacity : Int32

  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Bool || T == Int32 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported by Bottle" %}
    {% end %}
  end

  # Creates a new `Tensor` of an arbitrary size from a given
  # indexable *data*.  The type of the Tensor is inferred from
  # the provided data, as are the size, and stride.
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4, 5]
  #
  # v         # => Tensor[1, 2, 3, 4, 5]
  # typeof(v) # => Tensor(Int32)
  # ```
  def initialize(data : Indexable(T))
    check_type
    @size = data.size
    @buffer = Pointer(T).malloc(size) { |i| data[i] }
    @stride = 1
    @capacity = size.to_i
    @owner = true
  end

  def initialize(@buffer : Pointer(T), @size, @stride, @owner)
    check_type
    @capacity = @size * @stride
  end

  # Creates a new `Tensor` from a block.  Infers the type
  # of the Tensor from the value yielded by the block.
  #
  # ```
  # v = Tensor.new(5) { |i| i / 2 }
  #
  # v         # => Tensor[0.0, 0.5, 1.0, 1.5, 2.0]
  # typeof(v) # => Tensor(Float64)
  # ```
  def self.new(size : Int32, &block : Int32 -> T)
    buffer = Pointer(T).malloc(size) { |i| yield i }
    new(buffer, size, 1, true)
  end

  # Returns the element at the given *index*, without doing any bounds check.
  #
  # Make sure to invoke this method with *index* in `0...size`,
  # so converting negative indices to positive ones is not needed here.
  #
  # Clients never invoke this method directly. Instead, they access
  # elements with `#[](index)` and `#[]?(index)`.
  #
  # This method should only be directly invoked if you are absolutely
  # sure the index is in bounds, to avoid a bounds check for a small boost
  # of performance.
  @[AlwaysInline]
  def unsafe_at(index : Int)
    @buffer[index]
  end

  # Returns the element at the given index, if in bounds,
  # otherwise executes the given block and returns its value.
  #
  # ```
  # a = Tensor.new [1, 2, 3]
  # a.at(0) { nil } # => 1
  # a.at(2) { nil } # => nil
  # ```
  private def at(index : Int)
    index = check_index_out_of_bounds(index) do
      return yield
    end
    unsafe_at(index * @stride)
  end

  private def check_index_out_of_bounds(index)
    check_index_out_of_bounds(index) { raise IndexError.new }
  end

  private def check_index_out_of_bounds(index)
    index += size if index < 0
    if 0 <= index < size
      index
    else
      yield
    end
  end

  # Returns the element at the given index, if in bounds,
  # otherwise raises `IndexError`.
  #
  # ```
  # a = Tensor.new [1, 2, 3]
  # a[0] # => 1
  # a[5] # => IndexError
  # ```
  @[AlwaysInline]
  def [](index : Int)
    at(index) { raise IndexError.new }
  end

  # Sets the element at the given index, if in bounds,
  # otherwise raises `IndexError`.
  #
  # ```
  # a = Tensor.new [1, 2, 3]
  # a[0] = 100 # => 1
  # a[5] = 100 # => IndexError
  # ```
  @[AlwaysInline]
  def []=(index : Int, value : Number)
    index = check_index_out_of_bounds index
    @buffer[index * @stride] = T.new(value)
  end

  # Returns multiple elements identified by
  # a list of indexes. A copy is returned since
  # the memory is not guaranteed to be
  # contiguous.
  #
  # ```
  # a = Tensor.new [1, 2, 3, 4, 5]
  # a[[0, 2, 4]] # => Tensor[ 1  3  5]
  # ```
  def [](indexes : Indexable(Int))
    Tensor.new(indexes.size) { |i| self[i] }
  end

  # Set multiple elements at provided
  # indexes with an Indexable of values.
  #
  # ```
  # a = Tensor.new [1, 2, 3, 4, 5]
  # a[[0, 2, 4]] = [10, 10, 10]
  # ```
  def []=(indexes : Indexable(Int), values : Indexable(T))
    indexes.each_with_index { |e, i| self[e] = values[i] }
  end

  # Returns multiple elements identified by
  # a range. A view is returned since the
  # memory is contiguous.
  #
  # ```
  # a = Tensor.new [1, 2, 3, 4, 5]
  # a[2...] # => Tensor[ 3  4  5]
  # ```
  def [](range : Range(Int32?, Int32?))
    offset, count = Indexable.range_to_index_and_count(range, size)
    Tensor.new @buffer + offset * @stride, count, @stride, false
  end

  # Set multiple elements at a provide
  # range.
  #
  # ```
  # a = Tensor.new [1, 2, 3, 4, 5]
  # a[2...] = [10, 10, 10]
  # ```
  def []=(range : Range(Int32?, Int32?), values : Indexable(T))
    offset, count = Indexable.range_to_index_and_count(range, size)
    (offset...count).each_with_index { |e, i| self[e] = values[i] }
  end

  # Copies a Tensor, returns the copy.
  #
  # ```
  # a = Tensor.new [1, 2, 3]
  #
  # a.clone # => Tensor[ 1  2  3]
  # ```
  def clone
    Tensor(T).new(@capacity) { |i| @buffer[i] }
  end

  # Appends the string representation of
  # a `Tensor` to the provided *io*.
  #
  # ```
  # a = Tensor.new [1, 2, 3]
  #
  # a.clone # => Tensor[ 1  2  3]
  # ```
  def to_s(io)
    {% if T == Bool %}
      B::Util.vector_print(io, self, prefix = "Tensor[", override_max: true, maxval: false)
    {% else %}
      B::Util.vector_print(io, self)
    {% end %}
  end

  # Lazily yields the index values of a `Tensor`.  This method
  # is used as the core iteration method for `Tensors`
  #
  # ```crystal
  # v = Tensor.new [1, 2, 3, 4, 5]
  # v.each_index { |i| puts i }
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # # 4
  # ```
  def each_index(&block : Int32 -> _)
    0.step(to: @capacity - 1, by: @stride) do |n|
      yield n
    end
  end

  # Lazily yields a Tensor one element at a time.
  # The core iteration method for a Tensor
  #
  # ```crystal
  # v = Tensor.new [1, 2, 3, 4, 5]
  # v.each { |e| puts e }
  #
  # # 1
  # # 2
  # # 3
  # # 4
  # # 5
  # ```
  def each(&block)
    each_index do |i|
      yield @buffer[i]
    end
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
  def each_with_index(&block)
    each_index { |i| yield(@buffer[i], i) }
  end

  # Combines all elements in the Tensor by applying a binary operation, specified by a block, so as
  # to reduce them to a single value.
  #
  # For each element in the collection the block is passed an accumulator value (*memo*) and the element. The
  # result becomes the new value for *memo*. At the end of the iteration, the final value of *memo* is
  # the return value for the method. The initial value for the accumulator is the first element in the collection.
  def reduce(memo = T.new(0))
    each do |elem|
      memo = yield memo, elem
    end
    memo
  end

  # Initializes a Tensor with an uninitialized slice
  # of data.
  #
  # ```crystal
  # f = Tensor.empty(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def self.empty(n : Int32, dtype : U.class = Float64) forall U
    Tensor.new Pointer(U).malloc(n), n, 1, true
  end

  # Initializes a Tensor full of zeros.  Default
  # dtype is Float64, but other dtypes are supported.
  #
  # ```crystal
  # f = Tensor.zeros(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def self.zeros(n : Int32, dtype : U.class = Float64) forall U
    Tensor.new Pointer(U).malloc(n U.new(0)), n, 1, true
  end

  # Initializes a Tensor full of ones.  Default
  # dtype is Float64, but other dtypes are supported
  #
  # ```crystal
  # f = Tensor.ones(5, dtype: Int32)
  # f # => Tensor[1, 1, 1, 1, 1]
  # ```
  def self.ones(n : Int32, dtype : U.class = Float64) forall U
    Tensor.new Pointer(U).malloc(n, U.new(1)), n, 1, true
  end

  # Initializes a Tensor full of a given scalar.  Default
  # dtype is Float64, but other dtypes are supported
  #
  # ```crystal
  # f = Tensor.full(5, 5, dtype: Int32)
  # f # => Tensor[5, 5, 5, 5, 5]
  # ```
  def self.full(n : Int32, x : Number, dtype : U.class = Float64) forall U
    Tensor.new Pointer(U).malloc(n, U.new(x)), n, 1, true
  end

  # Pours a Tensor full of random data.
  # The dtype of the Tensor is inferred
  # from the values on either end of the
  # range.
  #
  # ```crystal
  # f = Tensor.random(0, 10, 5)
  # f # => Tensor[4, 8, 7, 8, 4]
  # ```
  def self.random(r : Range(U, U), n : Int32) forall U
    Tensor.new(n) { |_| Random.rand(r) }
  end

  # Casts a Tensor to another data dtype.
  # If the Tensor is already the given dtype,
  # a copy is not made, otherwise a new Tensor
  # is returned.
  #
  # ```crystal
  # f = Tensor.new [1, 2, 3]
  # f.astype(Float64) # => Tensor[1.0, 2.0, 3.0]
  # ```
  def astype(dtype : U.class) forall U
    if T == U
      return self
    end
    Tensor.new(self.size) { |i| U.new(self[i]) }
  end

  # Elementwise addition of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Tensor)
    B.add(self, other)
  end

  # Elementwise addition of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : Number)
    B.add(self, other)
  end

  # Elementwise subtraction of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Tensor)
    B.subtract(self, other)
  end

  # Elementwise subtraction of a Tensor with a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : Number)
    B.subtract(self, other)
  end

  # Elementwise multiplication of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Tensor)
    B.multiply(self, other)
  end

  # Elementwise multiplication of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : Number)
    B.multiply(self, other)
  end

  # Elementwise division of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Tensor)
    B.div(self, other)
  end

  # Elementwise division of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def /(other : Number)
    B.div(self, other)
  end

  # Computes the sum of each value of a Tensor
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.sum # => 10
  # ```
  def sum
    B.sum(self)
  end

  # Computes the maximum value of a Tensor
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    B.max(self)
  end

  # Computes the index of the maximum value of a Tensor
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    B.argmax(self)
  end

  # Computes the minimum value of a Tensor
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    B.min(self)
  end

  # Computes the index of the minimum value of a Tensor
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    B.argmin(self)
  end
end
