require "./flags"
require "./arrayprint"
require "./transform"
require "./stride_tricks"
require "../core/assemble"
require "../core/exceptions"
require "../iter/flat"
require "../iter/nd"
require "../iter/axes"
require "../iter/index"
require "../iter/permute"

abstract class Num::BaseArray(T)
  include Internal

  # Buffer pointing to the start of the array's data buffer
  getter buffer : Pointer(T)

  # Array of the array's dimensions
  property shape : Array(Int32)

  # Number of steps in each dimension to traverse the array
  property strides : Array(Int32)

  # Number of array dimensions
  getter ndims : Int32

  # Information about the memory layout of the array
  getter flags : ArrayFlags

  # The total number of elements in the array
  getter size : Int32

  # Base object if memory is from another array
  getter base : BaseArray(T)? = nil

  # All children must define a function that checks data for
  # valid dtypes.  For example, the Tensor class only holds
  # specific numeric types.
  abstract def check_type

  # The basetype of an array, useful for passing subclasses
  # through numeric methods, and needing to create new
  # instances of those subclsases.
  #
  # This is primarily used when self.class does not work, and
  # the same basetype but a different generic type must be returned.
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

  # Crates a scalar tensor, that acts like a scalar while still being
  # a Tensor.  This was primarily added so that indexing operations
  # could return single elements without having a union return type.
  def initialize(scalar : T)
    @buffer = Pointer(T).malloc(1, scalar)
    @ndims = 0
    @size = 1
    @shape = [] of Int32
    @strides = [] of Int32
    @flags = ArrayFlags::All
    @base = nil
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

  # Yields an Array from a proc, applying the function on each index of the Array
  def self.from_proc(shape : Array(Int32), prok : Proc(Int32, U), order : ArrayFlags = ArrayFlags::Contiguous) forall U
    total = shape.reduce { |i, j| i * j }
    ptr = Pointer(U).malloc(total) do |i|
      prok.call(i)
    end
    new(shape, order, ptr)
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

  # Returns a view of a Tensor from a list of indices or
  # ranges.
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

  def [](*args : Array(Int32))
    tp = args.to_a.transpose
    sliced = tp.map do |idx|
      slice_or_scalar(idx)
    end
    Assemble.stack(sliced)
  end

  private def slice_or_scalar(idx)
    if idx.is_a?(Array(Int32)) && idx.size == ndims
      return scalar(idx)
    end
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
  def []=(idx : Array, assign : BaseArray(T))
    can_write
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    old.flat_iter.zip(assign.flat_iter) do |i, j|
      i.value = j.value
    end
  end

  # Sets a view of an array from a list of indices or ranges
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

  # Slices a `Tensor` from an array of integers or ranges
  # Primarily used when you can't pass *args to the index
  # method but still need the functionality.
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  # t.slice([0, 0...1, 0...2]) #=>
  # Tensor([[0, 1]])
  # ```
  def slice(idx : Array)
    slice_from_indexers(idx)
  end

  # String representation of an n-dimensional array
  def to_s(io)
    io << "BaseArray(" << ArrayPrint.array2string(self, prefix: "BaseArray(") << ")"
  end

  # A flat iterator of the items in the array.  Always iterates
  # in c-contiguous order.
  def flat_iter
    if flags.contiguous?
      Iter::ContigFlatIter.new(self)
    else
      Iter::NDFlatIter.new(self)
    end
  end

  # A flat iterator with a flat index.
  def flat_iter_indexed
    i = -1
    flat_iter.each do |el|
      i += 1
      yield el, i
    end
  end

  # An iterator that never yields a STOP
  def unsafe_iter
    if flags.contiguous?
      Iter::UnsafeContigFlatIter.new(self)
    else
      Iter::UnsafeNDFlatIter.new(self)
    end
  end

  # Iterator along the indices of an array
  def index_iter
    IndexIter.new(shape)
  end

  def axis_iter(axis, keepdims = false)
    Iter::AxisIter.new(self, axis, keepdims)
  end

  def unsafe_axis_iter(axis, keepdims = false)
    Iter::UnsafeAxisIter.new(self, axis, keepdims)
  end

  # The value of an array's data pointer
  def value
    @buffer.value
  end

  # The type of elements contained in an array's data buffer, this ideally
  # should be a non-union data type.
  def dtype
    T
  end

  # The total size in bytes of each element of the array.  This is primarily
  # used for outputting an array to a .npy file, or reading a file from
  # an npy file into an array.
  def bytesize
    itemsize // sizeof(UInt8)
  end

  # The size, in bytes, of each element in an array.
  def itemsize
    {% if T == Bool %}
      1
    {% else %}
      sizeof(T)
    {% end %}
  end

  # Total number of bytes taken up by items in the `Tensor`s
  # data buffer.
  def nbytes
    size * sizeof(T)
  end

  # Duplicates a BaseArray, respecting the passed order of memory
  # provided.  Useful for throwing `Tensor`s down to LAPACK
  # since they must be in Fortran style order
  def dup(order : Char? = nil)
    Transform.dup(self, order)
  end

  # Shallow copies the `Tensor`.  Shape and strides are copied, but
  # the underlying data is not.  The returned `Tensor` does
  # not own its own data, and its base reflects that.
  def dup_view
    Transform.dup_view(self)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  def diag_view
    Transform.diag_view(self)
  end

  # New view of array with the same data.
  def view(dtype : U.class) forall U
    StrideTricks.view(self, dtype)
  end

  # Fits a `Tensor` into a new shape, not
  # altering memory if possible.  However, the `Tensor` is
  # usually copied.
  def reshape(newshape : Array(Int32))
    Transform.reshape(self, newshape)
  end

  # Fits a `Tensor` into a new shape, not
  # altering memory if possible.  However, the `Tensor` is
  # usually copied.
  def reshape(*args)
    Transform.reshape(self, *args)
  end

  # Flatten an array, returning a view if possible.
  # If the array is either fortran or c-contiguous, a view will be returned,
  def ravel
    Transform.ravel(self)
  end

  # Copy of the array, cast to a specified type.
  def astype(dtype : U.class) forall U
    Transform.astype(self, dtype)
  end

  # Permute the dimensions of a `Tensor`.  If no order is provided,
  # the dimensions will be reversed, a "true transpose".  Otherwise,
  # the dimensions will be permutated in the order provided.
  def transpose(order : Array(Int32))
    Transform.transpose(self, order)
  end

  # Permute the dimensions of a `Tensor`.  If no order is provided,
  # the dimensions will be reversed, a "true transpose".  Otherwise,
  # the dimensions will be permutated in the order provided.
  def transpose(*args)
    Transform.transpose(self, *args)
  end

  def tolist
    (0...shape[0]).map { |e| self[e] }
  end

  # Permutes an array along an axis, yielding the tensors at each
  # index of two varying axes.
  def permute_along_axis(axis)
    if ndims == 1
      yield self
    else
      if axis < 0
        axis = ndims + axis
      end
      raise "Axis out of range for this array" unless axis < ndims

      PermuteIter.new(self, axis).each do |perm|
        yield slice(perm)
      end
    end
  end

  # Reduces a tensor along an axis
  def reduce_fast(axis, keepdims = false)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims
    newshape = shape.dup
    newstrides = strides.dup
    ptr = buffer

    if !keepdims
      newshape.delete_at(axis)
      newstrides.delete_at(axis)
    else
      newshape[axis] = 1
      newstrides[axis] = 0
    end

    ret = self.class.new(buffer, newshape, newstrides, flags, nil).dup

    1.step(to: shape[axis] - 1) do |_|
      ptr += strides[axis]
      tmp = self.class.new(ptr, newshape, newstrides, flags, nil)
      ret.flat_iter.zip(tmp.flat_iter) do |x, y|
        yield x, y
      end
    end
    ret
  end

  def yield_along_axis(axis)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims
    newshape = shape.dup
    newstrides = strides.dup
    newshape.delete_at(axis)
    newstrides.delete_at(axis)
    ptr = buffer

    shape[axis].times do |_|
      tmp = self.class.new(ptr, newshape, newstrides, flags, nil)
      ptr += strides[axis]
      yield tmp
    end
  end

  # Apply an operation along the last axis of a tensor
  def apply_last_axis
    if ndims == 0
      raise Exceptions::ShapeError.new("Array has no axes")
    elsif ndims == 1
      yield self
    else
      n = shape[...-1].reduce { |i, j| i * j }
      newshape = shape[-1...]
      newstrides = strides[-1...]
      newbase = @base.nil? ? @base : self
      ptr = buffer
      0.step(to: n - 1) do |_|
        tmp = self.class.new(ptr, newshape, newstrides, ArrayFlags::None, newbase)
        yield tmp
        ptr += newstrides[0] * newshape[0]
      end
    end
  end

  # Accumulate an operation along an axis of a tensor
  def accumulate_fast(axis)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims
    arr = dup
    newshape = shape.dup
    newstrides = strides.dup
    ptr = arr.buffer
    newshape.delete_at(axis)
    newstrides.delete_at(axis)

    ret = self.class.new(buffer, newshape, newstrides, flags, nil)
    1.step(to: shape[axis] - 1) do |_|
      ptr += strides[axis]
      tmp = self.class.new(ptr, newshape, newstrides, flags, nil)
      tmp.flat_iter.zip(ret.flat_iter) do |ii, jj|
        yield ii, jj
      end
      ret = tmp
    end
    arr
  end

  # Until I figure out how to add broadcasting as an indexing operation,
  # this is how you expand the dimensions of a Tensor to make operations
  # easier to conduct on tensors with mismatching shapes.
  def bc(axis : Int32)
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

  def broadcastable(other : BaseArray)
    StrideTricks.broadcastable(self, other)
  end

  def broadcast_to(newshape : Array(Int32))
    StrideTricks.broadcast_to(self, newshape)
  end

  def as_strided(shape : Array(Int32), strides : Array(Int32), writeable = false)
    StrideTricks.as_strided(self, shape, strides, writeable)
  end

  protected def update_flags(flagmask : ArrayFlags = ArrayFlags::Contiguous, writeable = true)
    @flags = FlagChecks.update_flags(self, flags, flagmask, ndims, writeable)
  end

  # Computes the slice of an array from an array of indexers.
  private def slice_from_indexers(idx : Array)
    if idx.is_a?(Array(Int32)) && (idx.size == ndims)
      return scalar(idx)
    end
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
    newbase = @base ? @base : self

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
  private def aref_set(*args, value : BaseArray)
    idx = args.to_a
    fill = ndims - idx.size
    idx += [...] * fill
    old = slice_from_indexers(idx)
    if old.shape != value.shape
      value = value.broadcast_to(old.shape)
    end

    old.flat_iter.zip(value.flat_iter) do |i, j|
      i.value = T.new(j.value)
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
  private def aref_set(*args, value : Number)
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
    ptr = buffer
    ndims.times do |i|
      if strides[i] < 0
        ptr += (shape[i] - 1) * strides[i].abs
      end
    end

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
    self.class.new(ptr[offset])
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
      strides.zip(indexer, shape) do |i, j, k|
        if j < 0
          j += k
        end
        if j >= k
          raise IndexError.new("Index out of range")
        end
        offset += i * j
      end
      @buffer[offset] = T.new(value)
    end
  end

  # Calculates the shape of a standard library array
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

  # Checked on each attempt to write to an array, this value ensures that
  # arrays can only be edited if they are supposed to be, forbidding access
  # if for the example the array is a strided view where many elements share
  # the same memory locations.
  private def can_write
    unless flags.write?
      raise Exceptions::WriteError.new("Attempt to write to a read-only container")
    end
  end
end
