require "./nditer"
require "./indexer"
require "./ufunc"
require "benchmark"

@[Flags]
enum NDArray::ArrayFlags
  # Contiguous really means C-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The C-style
  # part means that the data is laid out such that the last index
  # is the fastest varying as one scans though the array's
  # memory.
  Contiguous
  # Fortran really means Fortran-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The Fortran-style
  # part means that the data is laid out such that the first index
  # is the fastest varying as one scans though the array's
  # memory.
  Fortran
  # OwnData indicates if this array is the owner of the data
  # pointed to by its .ptr property.  If not then this is a
  # view onto some other array's data.
  OwnData
end

struct NDArray::Tensor(T)
  # Unsafe pointer to a `Tensor`'s data.
  @buffer : Pointer(T)

  @base : Pointer(T)? = nil

  # Array-like container holding the dimensions of a `Tensor`
  getter shape : Array(Int32)

  # Array-like container holding the strides of a `Tensor`
  getter strides : Array(Int32)

  # Integer representing the number of dimensions of a `Tensor`
  getter ndims : Int32

  # Flags describing the memory layout of the array
  getter flags : ArrayFlags

  getter size : Int32

  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Bool || T == Int32 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported by Bottle" %}
    {% end %}
  end

  def self.new(_shape : Array(Int32), &block : Int32 -> U) forall U
    total = _shape.reduce { |i, j| i * j }
    ptr = Pointer(U).malloc(total) do |i|
      yield i
    end
    new(_shape, ArrayFlags::Contiguous, ptr)
  end

  def initialize(_shape : Array(Int32),
    order : ArrayFlags = ArrayFlags::Contiguous, ptr : Pointer(T)? = nil)
    @ndims = _shape.size

    # Empty NDArrays are both allowed, and they have
    # a shape of [0] and a stride of [1].  This
    # is inferred from an empty array being passed
    # to the constructor.
    if @ndims == 0
      @shape = [0]
      @strides = [1]

    # Otherwise, shape is directly copied from the
    # array that was passed to the constructor.
    # Strides will always share dimensionality
    # with the shape of an NDArray.
    else
      @shape = _shape.clone
      @strides = [0] * _shape.size
    end
    sz = 1

    # For empty arrays, since no elements are initialized
    # nothing special has to be done about the memory
    # allocation, but strides must be calculated differently.
    case order

    # For Fortran ordered arrays strides are calculated from
    # the beginning of the shape to the end, with strides
    # monotonically increasing.
    when ArrayFlags::Fortran
      @ndims.times do |i|
        @strides[i] = sz
        sz *= @shape[i]
      end

    # Otherwise, row major order is chosen and strides are
    # calculated from the reversed shape, monotonically
    # decreasing.
    else
      @ndims.times do |i|
        @strides[@ndims - i - 1] = sz
        sz *= @shape[@ndims - i - 1]
      end
    end

    @size = sz

    # Memory allocation for empty arrays is consistent
    # regardless of order, and this method will always
    # return an NDArray that owns its own data.
    @buffer = ptr.nil? ? Pointer(T).malloc(@size) : ptr
    @flags = order |= ArrayFlags::OwnData
  end

  def initialize(@buffer, @shape, @strides, @flags, @base)
    @ndims = @shape.size
    @size = @shape.reduce { |i, j| i * j }
  end

  def each_index(&block)
    indexes(shape.to_a) { |i| yield i }
  end

  def each(&block)
    each_index { |i| yield self[i] }
  end

  def each_with_index(&block)
    each_index { |i| yield self[i], i }
  end

  def flat
    FlatIter.new(self).each do |el|
      yield el
    end
  end

  def flat_iter
    FlatIter.new(self)
  end

  def unsafe_iter
    UnsafeIter.new(self)
  end

  def to_s(io)
    each_with_index do |el, i|
      io << startline(shape, i, ndims)
      {% if T == Bool %}
        io << "#{el}".rjust(7)
      {% else %}
        io << "#{el.round(3)}".rjust(3)
      {% end %}
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
    @buffer[offset]
  end

  def []=(indexer : Array(Int32), value : Number)
    offset = 0
    strides.zip(indexer) do |i, j|
      offset += i * j
    end
    @buffer[offset] = T.new(value)
  end

  # Returns a view of a NTensor from a list of indices or
  # ranges.
  #
  # ```
  # t = Tensor.new([2, 4, 4]) { |i| i }
  # ```
  def [](*args)
    # Setting up args to compute the offset, filling
    # missing dimensions with empty ranges so that
    # all ranges don't have to be explicitly defined
    idx = args.to_a
    fill = ndims - idx.size
    idx += [...] * fill

    # These will be mutated since the slice does
    # not necessarily share the shape of the base.
    newshape = shape.dup
    newstrides = strides.dup
    newflags = flags.dup

    newflags &= ~ArrayFlags::OwnData

    # Compute the new shape, strides and offset of the
    # data buffer.
    accessor = idx.map_with_index do |el, i|
      if el.is_a?(Int32)
        newshape[i] = 0
        newstrides[i] = 0
        el
      else
        start, offset = Indexable.range_to_index_and_count(el, shape[i])
        newshape[i] = offset
        start
      end
    end

    # Empty dimensions are collapsed from the `Tensor`
    newshape = newshape.reject { |i| i == 0 }
    newstrides = newstrides.reject { |i| i == 0 }
    newbase = @base ? @base : @buffer

    # Pointer is offset.
    ptr = @buffer
    strides.zip(accessor) do |i, j|
      ptr += i * j
    end

    # Create a tensor and update its flags since slicing can
    # disrupt continuity of the memory buffer.
    ret = Tensor(T).new(ptr, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::All)
    ret
  end

  def slice(idx : Array(Int32 |Range(Int32, Int32)))
    # These will be mutated since the slice does
    # not necessarily share the shape of the base.
    newshape = shape.dup
    newstrides = strides.dup
    newflags = flags.dup

    newflags &= ~ArrayFlags::OwnData

    # Compute the new shape, strides and offset of the
    # data buffer.
    accessor = idx.map_with_index do |el, i|
      if el.is_a?(Int32)
        newshape[i] = 0
        newstrides[i] = 0
        el
      else
        start, offset = Indexable.range_to_index_and_count(el, shape[i])
        newshape[i] = offset
        start
      end
    end

    # Empty dimensions are collapsed from the `Tensor`
    newshape = newshape.reject { |i| i == 0 }
    newstrides = newstrides.reject { |i| i == 0 }
    newbase = @base ? @base : @buffer

    # Pointer is offset.
    ptr = @buffer
    strides.zip(accessor) do |i, j|
      ptr += i * j
    end

    # Create a tensor and update its flags since slicing can
    # disrupt continuity of the memory buffer.
    ret = Tensor(T).new(ptr, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::All)
    ret
  end

  def dup(f : ArrayFlags::None)
    ret = Tensor(T).new(@shape)
    newcontig = f & (ArrayFlags::Fortran|ArrayFlags::Contiguous)
    contig = @flags & (ArrayFlags::Fortran|ArrayFlags::Contiguous)

    if contig && (newcontig == 0 || contig == newcontig)
      @buffer.copy_to(ret.@buffer, size)
    elsif ret.size
      if newcontig != (ret.flags & (ArrayFlags::Fortran|ArrayFlags::Contiguous))
        flat_iter.zip(ret.flat_iter) do |i, j|
          j.value = i.value
        end
        newstrides = ret.size
        ndims.times do |i|
          newstrides /= shape[i]
          ret.@strides[i] = newstrides
        end
      end
    end
    ret.update_flags(ArrayFlags::Fortran|ArrayFlags::Contiguous)
    ret
  end

  # Shallow copies the `Tensor`.  Shape and strides are copied, but
  # the underlying data is not.  The returned `Tensor` does
  # not own its own data, and its base reflects that.
  def dup_view
    newshape = @shape.dup
    newstrides = @strides.dup
    newflags = @flags.dup
    newflags &= ~ArrayFlags::OwnData
    Tensor(T).new(@buffer, newshape, newstrides, newflags, @buffer)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  def diag_view
    raise "Tensor must be two-dimensional" unless ndims == 2
    nel = Math.min(@shape[0], @shape[1])
    newshape = [nel]
    newstrides = [@strides[0] + @strides[1]]
    newflags = @flags.dup
    newbase = @base ? @base : @buffer
    ret = Tensor(T).new(@buffer, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::Fortran|ArrayFlags::Contiguous)
    ret
  end

  # Asserts if a `Tensor` is fortran contiguous, otherwise known
  # as stored in column major order.  This is not the default
  # layout for `Tensor`'s, but can provide performance benefits
  # when passing to LaPACK routines since otherwise the
  # `Tensor` must be transposed in memory.
  def is_fortran_contiguous
    # Empty Tensors are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return @shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    ndims.times do |i|
      dim = @shape[i]
      return true unless dim != 0
      return false unless @strides[i] == sd
      sd *= dim
    end
    true
  end

  # Asserts if a `Tensor` is c contiguous, otherwise known
  # as stored in row major order.  This is the default memory
  # storage for NDArray
  def is_contiguous
    # Empty Tensors are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    (ndims - 1).step(to: 0, by: -1) do |i|
      dim = @shape[i]
      return true unless dim != 0
      return false unless @strides[i] == sd
      sd *= dim
    end
    true
  end

  # Updates a `Tensor`'s flags by determining its
  # memory layout.  Multidimension tensors cannot be
  # both c and f contiguous, but this needs to be checked.
  #
  # This method should really only be called by internal
  # methods, or once stride tricks are exposed.
  protected def update_flags(flagmask)
    if flagmask & ArrayFlags::Fortran
      if is_fortran_contiguous
        @flags |= ArrayFlags::Fortran

        # mutually exclusive
        if ndims > 1
          @flags &= ~ArrayFlags::Contiguous
        end
      else
        @flags &= ~ArrayFlags::Fortran
      end
    end

    if flagmask & ArrayFlags::Contiguous
      if is_contiguous
        @flags |= ArrayFlags::Contiguous

        # mutually exclusive
        if ndims > 1
          @flags &= ~ArrayFlags::Fortran
        end
      else
        @flags &= ~ArrayFlags::Contiguous
      end
    end
  end

  def reduce_along_axis(axis)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for Tensor" unless axis < ndims

    idx0 = [0] * ndims
    idx1 = shape.dup
    idx1[axis] = 0

    ranges = idx0.zip(idx1).map_with_index do |i, idx|
      idx == axis ? 0 : i[0]...i[1]
    end
    ret = slice(ranges).dup

    1.step(to: shape[axis]-1) do |i|
      ranges[axis] = i
      slice(ranges).flat_iter.zip(ret.flat_iter) do |i, j|
        yield i, j
      end
    end
    ret
  end
end
