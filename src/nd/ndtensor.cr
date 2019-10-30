require "./dimension"
require "./strides"
require "./indexer"
require "./ufunc"
require "./numeric"

struct Bottle::NDTensor(T)
  # Unsafe pointer to a `Tensor`'s data.
  @ptr : Pointer(T)

  # Array-like contanier holding the dimensions of a `Tensor`
  getter shape : Shape

  # Array-like container holding the strides of a `Tensor`
  getter strides : Strides

  # Integer representing the number of dimensions of a `Tensor`
  getter ndims : Int32

  @contiguous : Bool = true

  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Bool || T == Int32 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported by Bottle" %}
    {% end %}
  end

  # Initializes a `Tensor` using a pointer to a buffer of valid
  # data, and a shape.  This is not a public facing method, and
  # should only be used by internal calls when a user has passed
  # valid data to another creation method, primarily `from_vector`
  #
  # ```
  # ptr = Pointer(Int32).malloc(4) { |i| i + 1 }
  #
  # t = NDTensor(ptr, Shape.new([2, 2]))
  # puts t # =>
  # # [[     1       2]
  # #  [     3       4]]
  # ```
  def initialize(@ptr : Pointer(T), @shape)
    check_type
    @ndims = shape.size
    @strides = shape.strides
  end

  # Initializes a `Tensor` using a pointer to a buffer of valid
  # data, and a shape.  This is not a public facing method, and
  # should only be used by internal calls when a user has passed
  # valid data to another creation method.
  #
  # Passing strides allows for strided subviews of NDTensors, and
  # so a creation method must exist that can handle them.
  def initialize(@ptr : Pointer(T), @shape, @strides)
    check_type
    @ndims = shape.size
  end

  # Initializes a `Tensor` from a provided block.  Primarily used for
  # operations of a `Tensor` that should return a copy, to allow
  # for simpler computation.
  #
  # ```
  # t = Tensor.new([2, 3]) { |i| i / 2 }
  # puts t # =>
  # # [[   0.0     0.5     1.0]
  # #  [   1.5     2.0     2.5]]
  # ```
  def self.new(shape : Array(Int32), &block : Int32 -> T)
    dims = Shape.new(shape)
    ptr = Pointer(T).malloc(dims.totalsize) do |i|
      yield i
    end
    new(ptr, dims)
  end

  def self.new(x : Shape, y : Shape, &block : Int32, Int32 -> T)
    dims = x + y
    t = y.totalsize
    ptr = Pointer(T).malloc(dims.totalsize) do |idx|
      i = idx // t
      j = idx % t
      yield i, j
    end
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

  def unsafe_flat_iter
    {@ptr, strides[-1]}
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

  def []=(indexer : Array(Int32), value : Number)
    offset = 0
    strides.zip(indexer) do |i, j|
      offset += i * j
    end
    @ptr[offset] = T.new(value)
  end

  # Returns a view of a NTensor from a list of indices or
  # ranges.  All dimensions must be explicitly provided currently
  # so that this overload will be chosen.  Otherwise, there
  # is possible ambiguity with the other indexer.
  #
  # ```
  # t = Tensor.new([2, 4, 4]) { |i| i }
  # ```
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

  # Elementwise addition of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : NDTensor)
    Bottle::NDimensional::UFunc.add(self, other)
  end

  # Elementwise addition of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : Number)
    Bottle::NDimensional::UFunc.add(self, other)
  end

  # Elementwise subtraction of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : NDTensor)
    Bottle::NDimensional::UFunc.subtract(self, other)
  end

  # Elementwise subtraction of a Tensor with a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : Number)
    Bottle::NDimensional::UFunc.subtract(self, other)
  end

  # Elementwise multiplication of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : NDTensor)
    Bottle::NDimensional::UFunc.multiply(self, other)
  end

  # Elementwise multiplication of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : Number)
    Bottle::NDimensional::UFunc.multiply(self, other)
  end

  # Elementwise division of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : NDTensor)
    Bottle::NDimensional::UFunc.div(self, other)
  end

  # Elementwise division of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def /(other : Number)
    Bottle::NDimensional::UFunc.div(self, other)
  end

  def transpose
    stridesnew = Strides.new(strides.dims.reverse)
    shapenew = Shape.new(shape.dims.reverse)
    NDTensor.new(@ptr, shapenew, stridesnew)
  end
end
