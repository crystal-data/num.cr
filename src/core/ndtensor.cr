require "./nditer"
require "./ufunc"
require "../iteration/iter"
require "../iteration/strategy"
require "../util/testing"
require "../util/exceptions"
require "./printoptions"
require "./statistics"

@[Flags]
enum Bottle::Internal::ArrayFlags
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

struct Bottle::Tensor(T)
  include Internal
  # Unsafe pointer to a `Tensor`'s data.
  @buffer : Pointer(T)

  # Pointer to the base data buffer of a `Tensor`'s data.
  # This will be not null if a `Tensor` does not own
  # its data, and is a view of another `Tensor`
  @base : Pointer(T)? = nil

  # Array-like container holding the dimensions of a `Tensor`
  getter shape : Array(Int32)

  # Array-like container holding the strides of a `Tensor`
  getter strides : Array(Int32)

  # Integer representing the number of dimensions of a `Tensor`
  getter ndims : Int32

  # Flags describing the memory layout of the array
  getter flags : ArrayFlags

  # The total number of elements contained in a `Tensor`
  getter size : Int32

  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Bool || T == Int32 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported by Bottle" %}
    {% end %}
  end

  # Yields a `Tensor` from a provided shape and a block.  The block only
  # provides the absolute index, not an index dependent on the shape,
  # so if a user wants to handle an arbitrary shape inside the block
  # they need to do that themselves.
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i / 2 }
  # t # =>
  # Tensor([[[ 0,  1],
  #          [ 2,  3]],
  #
  #         [[ 4,  5],
  #          [ 6,  7]],
  #
  #         [[ 8,  9],
  #          [10, 11]]])
  # ```
  def self.new(_shape : Array(Int32), order : ArrayFlags = ArrayFlags::Contiguous, &block : Int32 -> U) forall U
    total = _shape.reduce { |i, j| i * j }
    ptr = Pointer(U).malloc(total) do |i|
      yield i
    end
    new(_shape, order, ptr)
  end

  def self.from_array(_shape : Array(Int32), _data : Array)
    flat = _data.flatten
    ptr = flat.to_unsafe
    Testing.assert_compatible_shape(_shape, flat.size)
    if _shape.size == 0
      Tensor(typeof(flat[0])).new(_shape)
    else
      new(_shape) { |i| ptr[i] }
    end
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
      Tensor(T).new([] of Int32)
    else
      new([nrows, ncols], ArrayFlags::Contiguous, data)
    end
  end

  # A flexible method to create `Tensor`'s of arbitrary shapes
  # filled with random values of arbitrary types.  Since
  # Ranges can contain any dtype, the type of tensor is
  # inferred from the passed range, and a new `Tensor` is
  # sampled using the provided shape.
  #
  # ```
  # t = Tensor.random(0...10, [2, 2])
  # t # =>
  # Tensor([[5, 9],
  #         [3, 9]])
  # ```
  def self.random(r : Range(U, U), _shape : Array(Int32)) forall U
    if _shape.size == 0
      Tensor(U).new(_shape)
    else
      new(_shape) { |_| Random.rand(r) }
    end
  end

  # Creates a `Tensor` from a provided shape and order.
  # If a pointer to data is passed, no validation ensures
  # that the memory layout matches the passed order, so
  # this method is considered "unsafe".
  #
  # Primarily used by internal methods, however this can be used to
  # create empty `Tensors`, although the publicly
  # facing methods should be preferred.
  #
  # ```
  # t = Tensor(Int32).new([2, 2, 3], order: ArrayFlags::Fortran)
  # t # =>
  # Tensor([[[0, 0, 0],
  #          [0, 0, 0]],
  #
  #         [[0, 0, 0],
  #          [0, 0, 0]]])
  # ```
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
  def initialize(@buffer, @shape, @strides, @flags, @base)
    check_type
    @ndims = @shape.size
    @size = @shape.reduce { |i, j| i * j }
    update_flags(ArrayFlags::All)
  end

  # Returns a safe flattened view of the `Tensor.  This method will
  # raise `STOPITERATION` when the elements of the `Tensor`
  # have been exhausted.
  def flat_iter
    SafeNDIter.new(self).strategy
  end

  # Returns a safe transposed view of the `Tensor`.  Primarily
  # useful for swapping memory layout of an array in the duplication
  # routine.
  def trans_iter
    ContigIter.new(self, transpose: true)
  end

  # Returns an unsafe iter that does no bounds checking, but instead
  # keeps incrementing the data pointer of a `Tensor` indefinitely.
  #
  # Useful for fast iteration in internal methods, but is highly
  # unsafe for external API use.
  def unsafe_iter
    UnsafeNDIter.new(self).strategy
  end

  # Calculate the offset of an element in the `Tensor` from
  # a provided index and the strides of the `Tensor`
  def ptr_at(idx : Array(Int32))
    ptr = @buffer
    strides.zip(idx) do |i, j|
      ptr += i * j
    end
    ptr
  end

  # Creates a string representation of a `Tensor`.  The implementation
  # of this is a bit of a mess, but I am happy with the results currently,
  # it could however be cleaned up to handle long floating point values
  # more precisely.
  def to_s(io)
    printer = Internal::ToString::TensorPrint.new(self, io)
    printer.print
  end

  # An indexing method for an array of Integers
  # that produce a scalar.  I need to find a good
  # way to handle a case where the user provides
  # a list that is less than the dimensions of
  # the array, and needs to return an NTensor slice
  #
  # ```
  # a = Tensor.new([2, 3, 2]) { |i| i }
  # a[[0, 1, 0]] # => 8
  # ```
  def [](indexer : Array(Int32))
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
    @buffer[offset]
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
  def []=(indexer : Array(Int32), value : Number)
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
  # t = Tensor.new([2, 4, 4]) { |i| i }
  # ```
  def [](*args)
    # Setting up args to compute the offset, filling
    # missing dimensions with empty ranges so that
    # all ranges don't have to be explicitly defined
    idx = args.to_a
    fill = ndims - idx.size
    idx += [...] * fill
    slice_from_indexers(idx)
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
  def []=(idx : Array, assign : Tensor(T))
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    old.flat_iter.zip(assign.flat_iter) do |i, j|
      i.value = j.value
    end
  end

  def []=(*args : *U) forall U
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
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    old.flat_iter.each do |i|
      i.value = T.new(value)
    end
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
    ret = Tensor(T).new(ptr, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::All)
    ret
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
    ret = Tensor(T).new(shape, contig)
    if (contig & flags != ArrayFlags::None)
      ret = Tensor(T).new(shape, contig)
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
    Tensor(T).new(@buffer, newshape, newstrides, newflags, @buffer)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  #
  # ```
  # t = Tensor.new([3, 3]) { |i| i }
  # t.diag_view # => Tensor([0, 4, 8])
  def diag_view
    raise ShapeError.new("Tensor must be two-dimensional") unless ndims == 2
    nel = Math.min(@shape[0], @shape[1])
    newshape = [nel]
    newstrides = [@strides[0] + @strides[1]]
    newflags = @flags.dup
    newflags &= ~ArrayFlags::OwnData
    newbase = @base ? @base : @buffer
    ret = Tensor(T).new(@buffer, newshape, newstrides, newflags, newbase)
    ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
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
      ret = Tensor(T).new(@buffer, newshape, newstrides, newflags, newbase)
      ret.update_flags(ArrayFlags::Fortran | ArrayFlags::Contiguous)
      ret
    else
      tmp = self.dup
      ret = Tensor(T).new(tmp.@buffer, newshape, newstrides, tmp.flags.dup, nil)
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
      Tensor(T).new(@buffer, newshape, newstrides, newflags, newbase)
    elsif flags.fortran?
      newstrides = [strides[0]]
      newbase = @base ? @base : @buffer
      newflags &= ~ArrayFlags::OwnData
      Tensor(T).new(@buffer, newshape, newstrides, newflags, newbase)
    else
      reshape([-1])
    end
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
    ret = Tensor(T).new(@buffer, newshape, newstrides, @flags.dup, newbase)
    ret.update_flags(ArrayFlags::Contiguous | ArrayFlags::Fortran)
    ret
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

    1.step(to: shape[axis] - 1) do |i|
      ranges[axis] = i
      slice(ranges).flat_iter.zip(ret.flat_iter) do |ii, jj|
        yield ii, jj
      end
    end
    ret
  end

  # Iterates along the final two axes of a `Tensor`, useful
  # for matrix operations on n-dimensional Tensors.
  def matrix_iter
    MatrixIter.new(self)
  end

  # Total number of bytes taken up by items in the `Tensor`s
  # data buffer.
  def nbytes
    size * sizeof(T)
  end

  # Elementwise addition of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Tensor)
    Bottle::Internal::UFunc.add(self, other)
  end

  # Elementwise addition of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : Number)
    Bottle::Internal::UFunc.add(self, other)
  end

  # Elementwise subtraction of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Tensor)
    Bottle::Internal::UFunc.subtract(self, other)
  end

  # Elementwise subtraction of a Tensor with a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : Number)
    Bottle::Internal::UFunc.subtract(self, other)
  end

  # Elementwise multiplication of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Tensor)
    Bottle::Internal::UFunc.multiply(self, other)
  end

  # Elementwise multiplication of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : Number)
    Bottle::Internal::UFunc.multiply(self, other)
  end

  # Elementwise division of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Tensor)
    Bottle::Internal::UFunc.divide(self, other)
  end

  # Elementwise division of a Tensor to a scalar
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def /(other : Number)
    Bottle::Internal::UFunc.divide(self, other)
  end

  # Computes the sum of each value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # v.sum # => 10
  # ```
  def sum
    Bottle::Internal::Statistics.sum(self)
  end

  # Compuates the maximum value of a `Tensor`
  def max
    Bottle::Internal::Statistics.max(self)
  end
end
