require "./dimension"
require "./strides"
require "./indexer"
require "./ufunc"

struct NDTensor(T)
  @ptr : Pointer(T)
  getter shape : Shape
  getter strides : Strides
  getter ndims : Int32

  def initialize(@ptr : Pointer(T), @shape)
    @ndims = shape.size
    @strides = shape.strides
  end

  def initialize(@ptr : Pointer(T), @shape, @strides)
    @ndims = shape.size
  end

  def self.zeros(shape : Array(Int32), dtype : U.class = Float64) forall U
    dims = Shape.new(shape)
    ptr = Pointer(U).malloc(dims.totalsize, U.new(0))
    new(ptr, dims)
  end

  def self.new(shape : Array(Int32), &block : Int32 -> T)
    dims = Shape.new(shape)
    ptr = Pointer(T).malloc(dims.totalsize) do |i|
      yield i
    end
    new(ptr, dims)
  end

  def self.sequence(shape : Array(Int32), dtype : U.class = Float64) forall U
    dims = Shape.new(shape)
    ptr = Pointer(U).malloc(dims.totalsize) { |i| U.new(i) }
    new(ptr, dims)
  end

  def self.from_array(shape : Array(Int32), data : Array(U)) forall U
    dims = Shape.new(shape)
    ptr = Pointer(U).malloc(dims.totalsize) { |i| data[i] }
    new(ptr, dims)
  end

  def each_index(&block)
    indexes(shape.dims) { |i| yield i }
  end

  def each(&block)
    each_index { |i| yield self[i] }
  end

  def each_with_index(&block)
    each_index { |i| yield self[i], i }
  end

  def flat
    dims = Shape.new([shape.totalsize])
    stride = Strides.new([strides[-1]])
    NDTensor.new(@ptr, dims, stride)
  end

  def to_s(io)
    each_with_index do |el, i|
      io << startline(shape, i).rjust(ndims)
      io << "#{el.round(3)}".rjust(6)
      io << newline(shape, i)
    end
  end

  # An indexing method for an array of Integers
  # that produce a scalar.  I need to find a good
  # way to handle a case where the user provides
  # a list that is less than the dimensions of
  # the array, and needs to return an NTensor slice
  #
  # ```
  # a = NDArray.sequence([2, 3, 2])
  # a[[0, 1, 0]] # => 8.0
  # ```
  def [](indexer : Array(Int32))
    offset = 0
    strides.zip(indexer) do |i, j|
      offset += i * j
    end
    @ptr[offset]
  end

  # Returns a view of a NTensor from a list of indices or
  # ranges.
  def [](indexer : Array(Int32 | Range(Int32?, Int32?)))
    ranges, newshape, newstrides = index_to_range(indexer, shape, strides)

    start = 0

    strides.zip(ranges) do |i, j|
      start += i * j
    end

    NDTensor.new(
      (@ptr + start),
      Shape.new(newshape),
      Strides.new(newstrides)
    )
  end
end

include Bottle::Internal::UFunc
a = NDTensor.sequence([3, 3])
b = NDTensor.sequence([3, 3])

puts add.outer(a, b)
