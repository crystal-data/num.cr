require "./object"
require "../api/math"
require "../api/stats"
require "../api/vectorprint"
require "../ufunc/*"

class Tensor(T)
  # Returns the number of elements in the tensor
  #
  # ```
  # Tensor.new([1, 2, 3]).size # => 3
  # ```
  getter size : Int32

  getter stride : Int32

  # creates an empty Tensor
  def initialize
    @size = 0
    @stride = 1
    @buffer = Pointer(T).null
  end

  # Creates a new empty `Tensor` backed by a buffer that is
  # `size` big.
  #
  # Since `Tensor`'s have a fixed size, and only allocate their
  # data once, it is an efficient data structure for numerical
  # operations since memory rarely needs to be re-allocated
  #
  # ```
  # t = Tensor(Int32).new(5)
  # t.size # => 5
  # ```
  def initialize(capacity : Int)
    if capacity < 0
      raise ArgumentError.new("Negative tensor size: #{capacity}")
    end

    @stride = 1
    @size = capacity.to_i
    if capacity == 0
      @buffer = Pointer(T).null
    else
      @buffer = Pointer(T).malloc(capacity)
    end
  end

  # Creates a new `Tensor` of the given *size* filled with the a *value*
  #
  # ```
  # Tensor.new(3, 3.0) # => Tensor[3.0, 3.0, 3.0]
  # ```
  def initialize(size : Int, value : T)
    if size < 0
      raise ArgumentError.new("Negative tensor size: #{size}")
    end

    @size = size.to_i
    @stride = 1

    if size == 0
      @buffer = Pointer(T).null
    else
      @buffer = Pointer(T).malloc(size, value)
    end
  end

  def initialize(@buffer : Pointer(T), @size, @stride)
  end

  # Creates a new `Tensor` of the given *size* and invokes the given block once
  # for each index of `self`, assigning the block's value in that index.
  #
  # ```
  # Tensor.new(3) { |i| (i + 1) ** 2 } # => Tensor[1, 4, 9]
  # ```
  def self.new(size : Int, &block : Int32 -> T)
    Tensor(T).build(size) do |buffer|
      size.to_i.times do |i|
        buffer[i] = yield i
      end
      size
    end
  end

  # Creates a new `Tensor`, allocating an internal buffer with the given *capacity*,
  # and yielding that buffer. The given block must return the desired size of the tensor.
  #
  # This method is **unsafe**, but is usually used to initialize the buffer
  # by passing it to a C function.
  #
  # ```
  # Tensor.build(3) do |buffer|
  #   LibSome.fill_buffer_and_return_number_of_elements_filled(buffer)
  # end
  # ```
  def self.build(capacity : Int) : self
    tns = Tensor(T).new(capacity)
    tns.size = (yield tns.to_unsafe).to_i
    tns
  end

  # Returns a pointer to the internal buffer where `self`'s elements are stored.
  #
  # This method is **unsafe** because it returns a pointer, and the pointed might eventually
  # not be that of `self` if the internal buffer of the tensor is re-allocated
  #
  # ```
  # t = Tensor.new [1, 2, 3]
  # t.to_unsafe[0] # => 1
  # ```
  def to_unsafe : Pointer(T)
    @buffer
  end

  # :nodoc:
  protected def size=(size : Int)
    @size = size.to_i
  end

  # Sets the given value at the given index.
  #
  # Negative indices can be used to start counting from the end of the tensor.
  # Raises `IndexError` if trying to set an element outside the array's range.
  #
  # ```
  # tns = Tensor.new [1, 2, 3]
  # tns[0] = 5
  # p ary # => Tensor[5,2,3]
  #
  # tns[3] = 5 # raises IndexError
  # ```
  @[AlwaysInline]
  def []=(index : Int, value : T)
    index = check_index_out_of_bounds index
    @buffer[index * stride] = value
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

  # Returns all elements that are within the given range.
  #
  # Negative indices count backward from the end of the tensor (-1 is the last
  # element). Additionally, an empty tensor is returned when the starting index
  # for an element range is at the end of the tensor.
  #
  # Raises `IndexError` if the range's start is out of range.
  #
  # ```
  # t = Tensor.new [1, 2, 3, 4, 5, 6]
  # t[1..3]    # => [2, 3, 4]
  # t[4..7]    # => [5, 6]
  # t[6..10]   # raise IndexError
  # t[5..10]   # => []
  # t[-2...-1] # => [6]
  # t[2..]     # => [2, 3, 4, 5, 6]
  # ```
  def [](range : Range)
    start, offset = Indexable.range_to_index_and_count(range, size)
    Tensor.new @buffer + start, offset, stride
  end

  def to_s(io)
    io << @buffer.to_slice(size)
  end
end

t = Tensor.new(3) { |i| i + 1 }
m = t[1...]
m[1] = 8
puts t
