require "./flags"
require "./baseiter"
require "./print"
require "../util/exceptions"

abstract struct Bottle::BaseArray(T)
  include Internal
  @buffer : Pointer(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @ndims : Int32
  @flags : ArrayFlags
  @size : Int32
  @base : Pointer(T)? = nil

  getter shape
  getter strides
  getter ndims
  getter flags
  getter size
  getter buffer
  getter base

  setter shape
  setter strides

  private def can_write
    unless flags.write?
      raise Exceptions::WriteError.new("Attempt to write to a read-only Tensor")
    end
  end

  def dtype
    T
  end

  def bytesize
    sizeof(T) // sizeof(UInt8)
  end

  private def broadcast_strides(dest_shape, src_shape, dest_strides, src_strides)
    dims = dest_shape.size
    start = dims - src_shape.size

    ret = [0] * dest_strides.size
    (dims - 1).step(to: start, by: -1) do |i|
      s = src_shape[i - start]
      case s
      when 1
        ret[i] = 0
      when dest_shape[i]
        ret[i] = src_strides[i - start]
      else
        raise "Cannot broadcast from #{src_shape} to #{dest_shape}"
      end
    end
    ret
  end

  private def broadcast_equal(a, b)
    bc = true
    a.zip(b) do |i, j|
      if !(i == j || i == 1 || j == 1)
        bc = false
      end
    end
    bc
  end

  private def broadcastable_shape(a, b)
    a.zip(b).map do |i|
      Math.max(i[0], i[1])
    end
  end

  def broadcastable(other : BaseArray)
    return [] of Int32 unless shape != other.shape

    sz = shape.size
    osz = other.shape.size

    if sz == osz
      if broadcast_equal(shape, other.shape)
        return broadcastable_shape(shape, other.shape)
      end
    else
      if sz > osz
        othershape = [1] * (sz - osz) + other.shape
        if broadcast_equal(shape, othershape)
          return broadcastable_shape(shape, othershape)
        end
      else
        selfshape = [1] * (osz - sz) + shape
        if broadcast_equal(selfshape, other.shape)
          return broadcastable_shape(selfshape, other.shape)
        end
      end
    end
    raise Exceptions::ShapeError.new("Shapes #{shape} and #{other.shape} are not broadcastable")
  end

  def broadcast_to(newshape)
    dim = newshape.size
    defstrides = [0] * dim
    sz = 1
    dim.times do |i|
      defstrides[dim - i - 1] = sz
      sz *= newshape[dim - i - 1]
    end

    newstrides = broadcast_strides(newshape, shape, defstrides, strides)
    newflags = flags.dup
    newflags &= ~ArrayFlags::Contiguous
    newflags &= ~ArrayFlags::Fortran
    newflags &= ~ArrayFlags::Write
    newflags &= ~ArrayFlags::OwnData

    Tensor(T).new(@buffer, newshape, newstrides, newflags, @base, false)
  end

  # as_strided creates a view into the array given the exact strides and
  # shape. This means it manipulates the internal data structure of
  # a Tensor and, if done incorrectly, the array elements can point
  # to invalid memory and can corrupt results or crash your program.
  # It is advisable to always use the original x.strides when
  # calculating new strides to avoid reliance on a contiguous
  # memory layout.
  #
  # Furthermore, arrays created with this function often contain self
  # overlapping memory, so that two elements are identical.
  # Vectorized write operations on such arrays will typically be
  # unpredictable. They may even give different results for
  # small, large, or transposed arrays. Since writing to these
  # arrays has to be tested and done with great care, you may want
  # to use writeable=false to avoid accidental write operations.
  def as_strided(shape, strides, writeable = false)
    newflags = flags.dup
    if !writeable
      newflags = ArrayFlags::None
    end
    Tensor(T).new(@buffer, shape, strides, newflags, @base)
  end

  def bc?(axis : Int32)
    newshape = shape.dup
    newstrides = strides.dup
    if axis < ndims
      newshape.insert(axis, 1)
      newstrides.insert(axis, 0)
    elsif axis == ndims
      newshape << 1
      newstrides << 0
    else
      raise Exceptions::ShapeError.new("Too many dimensions for tensor")
    end
    as_strided(newshape, newstrides, true)
  end

  abstract def check_type
  abstract def basetype

  def initialize(_shape : Array(Int32),
                 order : ArrayFlags = ArrayFlags::Contiguous,
                 ptr : Pointer(T)? = nil)
    check_type
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
    @flags = order | ArrayFlags::OwnData
    update_flags(ArrayFlags::All)
  end

  # Internal method to create tensors from low level libraries.
  # This does no validation on inputs and is very unsafe unless
  # called by the library.
  #
  # Should not be used by the external API.
  def initialize(@buffer : Pointer(T), @shape, @strides, @flags, @base, update_flags = true)
    check_type
    @ndims = @shape.size
    @size = @shape.reduce { |i, j| i * j }
    if update_flags
      update_flags(ArrayFlags::All)
    end
  end

  # Yields a `BaseArray` from a provided shape and a block.  The block only
  # provides the absolute index, not an index dependent on the shape,
  # so if a user wants to handle an arbitrary shape inside the block
  # they need to do that themselves.
  #
  # ```
  # t = BaseArray.new([2, 2, 3]) { |i| i / 2 }
  # t # =>
  # Base([[[ 0,  1],
  #        [ 2,  3]],
  #
  #       [[ 4,  5],
  #        [ 6,  7]],
  #
  #       [[ 8,  9],
  #        [10, 11]]])
  # ```
  def self.new(shape : Array(Int32), order : ArrayFlags = ArrayFlags::Contiguous, &block : Int32 -> T)
    total = shape.reduce { |i, j| i * j }
    ptr = Pointer(T).malloc(total) do |i|
      yield i
    end
    new(shape, order, ptr)
  end

  # Yields a `Tensor` from a provided number of rows and columns.
  # This can quickly create matrices, useful for several `Tensor` creattion
  # methods such as the underlying implementation of `eye`, and `diag`.
  #
  # This method does provide *i* and *j* variables for the passed block,
  # so no offset calculations need to be done by the user.
  #
  # ```
  # t = Tensor.new(3, 3) { |i, j| i == j ? 1 : 0 }
  # t # =>
  # Tensor([[1, 0, 0],
  #         [0, 1, 0],
  #         [0, 0, 1]])
  # ```
  def self.new(nrows : Int32, ncols : Int32, &block : Int32, Int32 -> T)
    data = Pointer(T).malloc(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      yield i, j
    end
    if nrows == 0 && ncols == 0
      raise Exceptions::ShapeError.new("Cannot initialize an empty matrix")
    else
      new([nrows, ncols], ArrayFlags::Contiguous, data)
    end
  end

  private def self.calculate_shape(arr, calc_shape : Array(Int32) = [] of Int32)
    return calc_shape unless arr.is_a?(Array)

    first_el = arr[0]
    if first_el.is_a?(Array)
      lc = first_el.size
      unless arr.all? do |el|
               el.is_a?(Array) && el.size == lc
             end
        raise Exceptions::ShapeError.new("All subarrays must be the same length")
      end
    end

    calc_shape << arr.size
    calc_shape = calculate_shape(arr[0], calc_shape)
    calc_shape
  end

  def self.from_array(array : Array)
    newshape = calculate_shape(array)
    dims = newshape.size
    newstrides = [0] * dims

    sz = 1

    dims.times do |i|
      newstrides[dims - i - 1] = sz
      sz *= newshape[dims - i - 1]
    end

    ptr = array.flatten.to_unsafe

    new(ptr, newshape, newstrides, ArrayFlags::Contiguous, nil)
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
  protected def update_flags(flagmask, writeable = true)
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

    if writeable
      @flags |= ArrayFlags::Write
    end
  end

  def to_s(io)
    printer = ToString::BasePrinter.new(self, io)
    printer.print
  end

  def flat_iter
    SafeNDIter.new(self).strategy
  end

  def unsafe_iter
    UnsafeNDIter.new(self).strategy
  end

  def index_iter
    IndexIter.new(shape)
  end

  private def slice_from_indexers(idx : Array)
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
        if el < 0
          el += shape[i]
        end
        if el >= shape[i]
          raise IndexError.new("Index out of range")
        end
        el
      elsif el.is_a?(Range)
        start, offset = Indexable.range_to_index_and_count(el, shape[i])
        if start >= shape[i]
          raise IndexError.new("Index out of range")
        end
        newshape[i] = offset
        start
      elsif el.is_a?(Tuple)
        range, step = el
        abstep = step.abs
        start, offset = Indexable.range_to_index_and_count(range, shape[i])
        newshape[i] = offset // abstep + offset % abstep
        newstrides[i] = step * newstrides[i]
        start
      else
        start, offset = Indexable.range_to_index_and_count(..., shape[i])
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
    ret = self.class.new(ptr, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::All)
    ret
  end

  # An indexing method for an array of Integers
  # that produce a scalar. To disallow ambiguity
  # of return type, this method must be passed
  # as a list.
  #
  # ```
  # a = Tensor.new([2, 3, 2]) { |i| i }
  # a[[0, 1, 0]] # => 8
  # ```
  private def scalar(indexer : Array(Int32))
    offset = 0
    strides.zip(indexer, shape) do |i, j, k|
      if j < 0
        j += k
      end
      if j >= k
        raise IndexError.new("Index out of range")
      end
      offset += i * j
    end
    Tensor(T).new([1]) { |_| @buffer[offset] }
  end

  def value
    @buffer.value
  end

  # Sets a single value in a `Tensor` based on
  # the provided index.  Casting will occur
  # so that the number matches the type of the
  # Tensor.
  #
  # ```
  # a = B.arange(10)
  # a[[1]] = 100
  # a # => Tensor([  0, 100,   2,   3,   4,   5,   6,   7,   8,   9])
  # ```
  private def scalar_set(indexer : Array(Int32), value : Number)
    can_write
    if indexer.size < strides.size
      fill = ndims - indexer.size
      indexer += [...] * fill
      old = slice_from_indexers(indexer)
      old.flat_iter.each do |i|
        i.value = T.new(value)
      end
    else
      offset = 0
      strides.zip(indexer) do |i, j|
        offset += i * j
      end
      @buffer[offset] = T.new(value)
    end
  end

  # Returns a view of a NTensor from a list of indices or
  # ranges.
  #
  # ```
  # t = BaseArray.new([2, 4, 4]) { |i| i }
  # ```
  def [](*args)
    # Setting up args to compute the offset, filling
    # missing dimensions with empty ranges so that
    # all ranges don't have to be explicitly defined
    idx = args.to_a
    if idx.is_a?(Array(Int32)) && idx.size == ndims
      return scalar(idx)
    end
    fill = ndims - idx.size
    idx += [...] * fill
    slice_from_indexers(idx)
  end

  def [](mask : Tensor(Bool))
    if mask.shape != shape
      mask = mask.broadcast_to(shape)
    end

    ret = Pointer(T).malloc(size)
    elems = 0
    flat_iter.zip(mask.flat_iter) do |i, j|
      if j.value
        ret[elems] = i.value
        elems += 1
      end
    end
    ret = ret.realloc(elems)
    Tensor(T).new([elems]) { |i| ret[i] }
  end

  # Assigns a `Tensor` to a slice of an array.
  # The provided tensor must be the same shape
  # as the slice in order for this method to
  # work.
  #
  # ```
  # t = Tensor.new([3, 2, 2]) { |i| i }
  # t[[1]] = Tensor.new([2, 2]) { |i| i * 20 }
  # t # =>
  # Tensor([[[ 0,  1],
  #          [ 2,  3]],
  #
  #         [[ 0, 20],
  #          [40, 60]],
  #
  #         [[ 8,  9],
  #          [10, 11]]])
  # ```
  def []=(idx : Array, assign : BaseArray(T))
    can_write
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    old.flat_iter.zip(assign.flat_iter) do |i, j|
      i.value = j.value
    end
  end

  def []=(*args : *U) forall U
    can_write
    {% begin %}
      aref_set(
        {% for i in 0...U.size - 1 %}
          args[{{i}}],
        {% end %}
        value: args[{{U.size - 1}}]
      )
    {% end %}
  end

  # Assigns a `Tensor` to a slice of an array.
  # The provided tensor must be the same shape
  # as the slice in order for this method to
  # work.
  #
  # ```
  # t = Tensor.new([3, 2, 2]) { |i| i }
  # t[[1]] = Tensor.new([2, 2]) { |i| i * 20 }
  # t # =>
  # Tensor([[[ 0,  1],
  #          [ 2,  3]],
  #
  #         [[ 0, 20],
  #          [40, 60]],
  #
  #         [[ 8,  9],
  #          [10, 11]]])
  # ```
  def aref_set(*args, value : Tensor(T))
    idx = args.to_a
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    if old.shape != value.shape
      value = value.broadcast_to(old.shape)
    end

    old.flat_iter.zip(value.flat_iter) do |i, j|
      i.value = j.value
    end
  end

  # Assigns a scalar value to a slice of a `Tensor`.
  # The value is tiled along the entire slice.
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  # t[[1]] = 99
  # t #=>
  # Tensor([[[ 0,  1],
  #          [ 2,  3]],
  #
  #         [[99, 99],
  #          [99, 99]],
  #
  #         [[99, 99],
  #          [99, 99]]])
  # ```
  def aref_set(*args, value : Number)
    idx = args.to_a
    if idx.is_a?(Array(Int32)) && idx.size == ndims
      scalar_set(idx, value)
    else
      fill = ndims - idx.size
      idx += [...] * fill
      old = slice_from_indexers(idx)
      old.flat_iter.each do |i|
        i.value = T.new(value)
      end
    end
  end

  # Slices a `Tensor` from an array of integers or ranges
  # Primarily used when you can't pass *args to the index
  # method but still need the functionality.
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  # t.slice([0, 0...1, 0...2]) #=>
  # Tensor([[0, 1]])
  # ```
  def slice(idx : Array(Int32 | Range(Int32, Int32)))
    slice_from_indexers(idx)
  end

  # Duplicates a `Tensor`, respecting the passed order of memory
  # provided.  Useful for throwing `Tensor`s down to LAPACK
  # since they must be in Fortran style order
  #
  # ```
  # t = B.arange(5)
  # t.dup # => Tensor([0, 1, 2, 3, 4])
  # ```
  def dup(order : Char? = nil) forall U
    contig = uninitialized ArrayFlags
    case order
    when 'C'
      contig = ArrayFlags::Contiguous
    when 'F'
      contig = ArrayFlags::Fortran
    when nil
      contig = flags & (ArrayFlags::Contiguous | ArrayFlags::Fortran)
    else
      raise Exceptions::ValueError.new(
        "Invalid argument for order.  Valid options or 'C', or 'F'")
    end
    ret = self.class.new(shape, contig)
    if (contig & flags != ArrayFlags::None)
      ret = self.class.new(shape, contig)
      @buffer.copy_to(ret.@buffer, size)
    else
      ret.flat_iter.zip(flat_iter).each do |i, j|
        i.value = j.value
      end
    end
    ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
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
    self.class.new(@buffer, newshape, newstrides, newflags, @buffer)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  #
  # ```
  # t = Tensor.new([3, 3]) { |i| i }
  # t.diag_view # => Tensor([0, 4, 8])
  def diag_view
    raise Exceptions::ShapeError.new("Tensor must be two-dimensional") unless ndims == 2
    nel = Math.min(@shape[0], @shape[1])
    newshape = [nel]
    newstrides = [@strides[0] + @strides[1]]
    newflags = @flags.dup
    newflags &= ~ArrayFlags::OwnData
    newbase = @base ? @base : @buffer
    ret = self.class.new(@buffer, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
    ret
  end

  # Fits a `Tensor` into a new shape, not
  # altering memory if possible.  However, the `Tensor` is
  # usually copied.
  #
  # ```
  # t = Tensor.new([2, 4, 3]])
  #
  # t.reshape([2, 2, 2, 3]) # =>
  # Tensor([[[[ 0,  1,  2],
  #           [ 6,  7,  8]],
  #
  #          [[ 3,  4,  5],
  #           [ 9, 10, 11]]],
  #
  #
  #         [[[12, 13, 14],
  #           [18, 19, 20]],
  #
  #          [[15, 16, 17],
  #           [21, 22, 23]]]])
  # ```
  def reshape(newshape : Array(Int32))
    if newshape == @shape
      return self
    end
    newsize = 1
    cur_size = size
    autosize = -1
    newshape.each_with_index do |val, i|
      if val < 0
        if autosize >= 0
          raise Exceptions::ValueError.new("Only shape dimension can be automatic")
        end
        autosize = i
      else
        newsize *= val
      end
    end

    if autosize >= 0
      newshape = newshape.dup
      newshape[autosize] = cur_size // newsize
      newsize *= newshape[autosize]
    end

    if newsize != cur_size
      raise "Shapes #{@shape} cannot be reshaped to #{newshape}"
    end

    stride = uninitialized Int32
    newstrides = [0] * newshape.size
    newbase = @base ? @base : @buffer
    newdims = newshape.size

    if @flags & ArrayFlags::Contiguous
      stride = 1
      newdims.times do |i|
        newstrides[newdims - i - 1] = stride
        stride *= newshape[newdims - i - 1]
      end
    else
      stride = 1
      newshape.each_with_index do |d, i|
        newstrides[i] = stride
        stride *= d
      end
    end

    if flags.fortran? || flags.contiguous?
      newflags = @flags.dup
      newflags &= ~ArrayFlags::OwnData
      ret = self.class.new(@buffer, newshape, newstrides, newflags, newbase)
      ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
      ret
    else
      tmp = self.dup
      ret = self.class.new(tmp.@buffer, newshape, newstrides, tmp.flags.dup, nil)
      ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
      ret
    end
  end

  # Flatten an array, returning a view if possible.
  # If the array is either fortran or c-contiguous, a view will be returned,
  #
  # otherwise, the array will be reshaped and copied.
  def ravel
    newshape = [size]
    newflags = flags.dup
    if flags.contiguous?
      newstrides = [strides[-1]]
      newbase = @base ? @base : @buffer
      newflags &= ~ArrayFlags::OwnData
      self.class.new(@buffer, newshape, newstrides, newflags, newbase)
    elsif flags.fortran?
      newstrides = [strides[0]]
      newbase = @base ? @base : @buffer
      newflags &= ~ArrayFlags::OwnData
      self.class.new(@buffer, newshape, newstrides, newflags, newbase)
    else
      reshape([-1])
    end
  end

  def astype(dtype : U.class) forall U
    ret = Tensor(U).new(shape)
    {% if U == Bool %}
      ret.flat_iter.zip(flat_iter) do |i, j|
        i.value = (j.value != 0) && (!!j.value)
      end
    {% else %}
      ret.flat_iter.zip(flat_iter) do |i, j|
        i.value = U.new(j.value)
      end
    {% end %}
    ret
  end

  def view(dtype : U.class) forall U
    if T == U
      return dup_view
    end

    tsize = T == Bool ? 1 : sizeof(T)
    usize = U == Bool ? 1 : sizeof(U)

    newsize = tsize / usize
    newlast = shape[-1] * newsize
    intlast = Int32.new newlast

    if newlast != intlast || intlast == 0
      raise "Cannot view array as #{U}"
    end

    newshape = shape.dup
    newstrides = strides.dup
    newflags = flags.dup
    ptr = @buffer.unsafe_as(Pointer(U))

    newshape[-1] = Int32.new(newlast)

    sz = 1
    @ndims.times do |i|
      newstrides[@ndims - i - 1] = sz
      sz *= newshape[@ndims - i - 1]
    end

    Tensor(U).new(ptr, newshape, newstrides, newflags, nil)
  end

  # Permute the dimensions of a `Tensor`.  If no order is provided,
  # the dimensions will be reversed, a "true transpose".  Otherwise,
  # the dimensions will be permutated in the order provided.
  #
  # ```
  # t = Tensor.new([2, 4, 3]) { |i| i }
  # t.transpose([2, 0, 1])
  # Tensor([[[ 0,  3,  6,  9],
  #          [12, 15, 18, 21]],
  #
  #         [[ 1,  4,  7, 10],
  #          [13, 16, 19, 22]],
  #
  #         [[ 2,  5,  8, 11],
  #          [14, 17, 20, 23]]])
  # ```
  def transpose(order : Array(Int32) = [] of Int32)
    newshape = @shape.dup
    newstrides = @strides.dup
    newbase = @base ? @base : @buffer
    if order.size == 0
      order = (0...ndims).to_a.reverse
    end
    n = order.size
    if n != ndims
      raise "Axes don't match array"
    end

    permutation = [0] * 32
    reverse_permutation = [0] * 32
    n.times do |i|
      reverse_permutation[i] = -1
    end

    n.times do |i|
      axis = order[i]
      if axis < 0
        axis = ndims + axis
      end
      if axis < 0 || axis >= ndims
        raise "Invalid axis for this array"
      end
      if reverse_permutation[axis] != -1
        raise "Repeated axis in transpose"
      end
      reverse_permutation[axis] = i
      permutation[i] = axis
    end

    n.times do |i|
      newshape[i] = @shape[permutation[i]]
      newstrides[i] = strides[permutation[i]]
    end
    ret = self.class.new(@buffer, newshape, newstrides, @flags.dup, newbase)
    ret.update_flags(ArrayFlags::Contiguous | ArrayFlags::Fortran)
    ret
  end

  def reduce_along_axis(axis)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims

    idx0 = [0] * ndims
    idx1 = shape.dup
    idx1[axis] = 0

    ranges = idx0.zip(idx1).map_with_index do |i, idx|
      idx == axis ? 0 : i[0]...i[1]
    end
    ret = slice(ranges).dup

    1.step(to: shape[axis] - 1) do |i|
      ranges[axis] = i
      slice(ranges).flat_iter.zip(ret.flat_iter) do |ii, jj|
        yield ii, jj
      end
    end
    ret
  end

  def accumulate_along_axis(axis)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims

    idx0 = [0] * ndims
    idx1 = shape.dup
    idx1[axis] = 0

    arr = dup

    ranges = idx0.zip(idx1).map_with_index do |i, idx|
      idx == axis ? 0 : i[0]...i[1]
    end

    ret = arr.slice(ranges)

    1.step(to: shape[axis] - 1) do |i|
      ranges[axis] = i
      newret = arr.slice(ranges)
      newret.flat_iter.zip(ret.flat_iter) do |ii, jj|
        yield ii, jj
      end
      ret = newret
    end
    arr
  end

  # Total number of bytes taken up by items in the `Tensor`s
  # data buffer.
  def nbytes
    size * sizeof(T)
  end
end
