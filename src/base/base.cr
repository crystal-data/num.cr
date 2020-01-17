require "./flags"
require "../print"
require "./transform"
require "./stride_tricks"
require "../core/assemble"
require "../core/exceptions"
require "../iter/flat"
require "../iter/nd"
require "../iter/axes"
require "../iter/index"
require "../iter/permute"
require "../iter/macros"
require "./helpers"
require "./broadcast"

# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
class Num::BaseArray(T)
  # Buffer pointing to the start of the array's data buffer
  getter buffer : Pointer(T)

  # Array of the array's dimensions
  getter shape : Array(Int32)

  # Number of steps in each dimension to traverse the array
  getter strides : Array(Int32)

  # Number of array dimensions
  getter ndims : Int32

  # Information about the memory layout of the array
  getter flags : NumInternal::ArrayFlags

  # The total number of elements in the array
  getter size : Int32

  def basetype
    BaseArray(T)
  end

  def basetype(klass : U.class) forall U
    BaseArray(U)
  end

  # This should be overwritten by all base classes.
  # Determines if an array is initialized containing
  # a valid type
  def type?
  end

  # Checks the write flag on an array, raising a WriteError
  # if the array is read-only, such as an array returned
  # from a method like as_strided
  def write?
    if !flags.write?
      raise NumInternal::WriteError.new("Array is read-only")
    end
  end

  # Initialization method for a generic buffer, shape and strides.
  # This method updates flags on the passed array, if the array
  # is read only, its flags need to be updated later
  def initialize(@buffer : Pointer(T), @shape, @strides, @flags = NumInternal::ArrayFlags::All)
    type?
    @size = @shape.product
    @ndims = @shape.size
    update_flags
  end

  # Initializes an ndarray from a provided shape and memory layout.
  # Flags will be auto-computed
  def initialize(@shape : Array(Int32), order : Char = 'C')
    type?
    @ndims = @shape.size

    if @ndims == 0
      @shape = [0]
      @strides = [1]
    else
      @strides = NumInternal.shape_to_strides(@shape, order)
    end

    @size = @shape.product
    @buffer = Pointer(T).malloc(@size)
    @flags = NumInternal::ArrayFlags::All
    update_flags
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
    @flags = NumInternal::ArrayFlags::All
  end

  # Yields an array from a provided shape and a block.  The block only
  # provides the absolute index, not an index dependent on the shape,
  # so if a user wants to handle an arbitrary shape inside the block
  # they need to do that themselves.
  def self.new(shape : Array(Int32), order : Char = 'C', &block : Int32 -> T)
    total = shape.product
    ptr = Pointer(T).malloc(total) do |i|
      yield i
    end
    strides = NumInternal.shape_to_strides(shape, order)
    new(ptr, shape, strides)
  end

  # Yields an array from a provided number of rows and columns.
  # This can quickly create matrices, useful for several `Tensor` creattion
  # methods such as the underlying implementation of `eye`, and `diag`.
  #
  # This method does provide *i* and *j* variables for the passed block,
  # so no offset calculations need to be done by the user.
  def self.new(nrows : Int32, ncols : Int32, &block : Int32, Int32 -> T)
    data = Pointer(T).malloc(nrows * ncols) do |idx|
      i = idx // ncols
      j = idx % ncols
      yield i, j
    end
    if nrows == 0 || ncols == 0
      raise NumInternal::ShapeError.new("Cannot initialize an empty matrix")
    end
    strides = NumInternal.shape_to_strides([nrows, ncols], 'C')
    new(data, [nrows, ncols], strides)
  end

  def self.from_array(array : Array)
    newshape = calculate_shape(array)
    dims = newshape.size
    newstrides = NumInternal.shape_to_strides(newshape)
    ptr = array.flatten.to_unsafe
    new(ptr, newshape, newstrides)
  end

  def self.from_proc(shape : Array(Int32), prok : Proc(Int32, U), order : Char = 'C') forall U
    size = shape.product
    ptr = Pointer(U).malloc(size) do |i|
      prok.call(i)
    end
    new(ptr, shape, NumInternal.shape_to_strides(shape, order))
  end

  # Shallow copies the `Tensor`.  Shape and strides are copied, but
  # the underlying data is not.  The returned `Tensor` does
  # not own its own data, and its base reflects that.
  def dup_view
    newshape = @shape.dup
    newstrides = @strides.dup
    newflags = @flags.dup
    newflags &= ~NumInternal::ArrayFlags::OwnData
    self.class.new(@buffer, newshape, newstrides, newflags)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  def diag_view
    raise NumInternal::ShapeError.new("Array must be two-dimensional") unless ndims == 2
    nel = @shape.min
    newshape = [nel]
    newstrides = [@strides.sum]
    newflags = @flags.dup
    newflags &= ~NumInternal::ArrayFlags::OwnData
    ret = self.class.new(@buffer, newshape, newstrides, newflags)
    ret.update_flags(NumInternal::ArrayFlags::Fortran | NumInternal::ArrayFlags::Contiguous)
    ret
  end

  # Returns a view of an ndarray from a list of indexers.
  # Valid indexers are integers, such as `1` or `2`, ranges, such
  # as `...` or `1...`, and tuples of a range, and an integer representing
  # the step of the index operation, such as `{..., -1}`
  def [](*args)
    slice_internal(args.to_a)
  end

  def [](idx : Array)
    slice_internal(idx)
  end

  # Writes to a view of an ndarray given a variadic tuple of indexers and a value to
  # map across the array.  The value to map will always be the last value
  # provided as args, so with some macro magic we can defer to the correct
  # method
  def []=(*args : *U) forall U
    write?
    {% begin %}
      aref_set(
        {% for i in 0...U.size - 1 %}
          args[{{i}}],
        {% end %}
        value: args[{{U.size - 1}}]
      )
    {% end %}
  end

  # Writes to a view of an ndarray given a list of indexers and a value to
  # map across the array.  This is equivalent to the variadic method in
  # actual functionality
  def []=(idx : Array, assign : BaseArray)
    write?
    old = self.slice(idx)
    old.map2!(assign) do |_, j|
      {% if T == String %}
        j.to_s
      {% elsif T == Bool %}
        j != 0 ? true : false
      {% else %}
        T.new(j)
      {% end %}
    end
  end

  # For anything passed other than a BaseArray, this method treats it as
  # a scalar and maps it across the resulting slice.
  def []=(idx : Array, assign : U) forall U
    write?
    old = self.slice(idx)
    old.map! { |_| T.new(assign) }
  end

  # Explicit method for slicing an array using a list of indexers, while
  # it could probably be done with
  def slice(args : Array)
    slice_internal(args)
  end

  def value
    @buffer.value
  end

  def astype(dtype : U.class) forall U
    ret = self.basetype(U).new(@shape)
    {% if U == Bool %}
      ret.iter.zip(iter) do |i, j|
        i.value = (j.value != 0) && (!!j.value)
      end
    {% else %}
      ret.iter.zip(iter) do |i, j|
        i.value = U.new(j.value)
      end
    {% end %}
    ret
  end

  # Returns an iterator over the flattened elements of an array.
  # This iterator will always iterate in C-contiguous order, regardless
  # of the strides of the array.  It will iterate much faster over
  # arrays that are c-contiguous
  def iter
    if flags.contiguous?
      NumInternal::ContigFlatIter.new(self)
    else
      NumInternal::NDFlatIter.new(self)
    end
  end

  def unsafe_iter
    if flags.contiguous?
      NumInternal::UnsafeContigFlatIter.new(self)
    else
      NumInternal::UnsafeNDFlatIter.new(self)
    end
  end

  def axis(axis = -1, keepdims = false)
    NumInternal::AxisIter.new(self, axis, keepdims)
  end

  # Adds the string representation of an ndarray to the passed IO
  def to_s(io)
    io << to_s
  end

  # Returns a string representation of an ndarray
  def to_s
    NumInternal.array_to_string(self)
  end

  # Adds the string representation of an ndarray to the passed IO, this
  # is primarily useful for informing how the array looks when stored
  # in an array
  def inspect(io)
    to_s(io)
  end

  # Maps an ndarray in place, storing the result of the yielded block.
  # Since this is an in-place operation, the result is cast to the type
  # of the input, and information may be lost if the operation otherwise
  # would have required a cast.
  def map!
    iter.each do |i|
      i.value = T.new(yield i.value)
    end
  end

  # Maps an ndarray and returns a copy, storing the result of the yielded block.
  # This can result in a different return type, since the operation allocates
  # a new array before returning, so no casting will occur back to the original
  # dtype
  def map(&block : T -> U) forall U
    ret = basetype(U).new(shape, 'C')
    NumInternal::NDFlatIter2.new(ret, self).each do |i, j|
      i.value = yield j.value
    end
    ret
  end

  # Maps a function across two ndarrays.  This is an inplace operation, so the
  # shape of the other array must be broadcastable to the shape of the input
  # array.
  def map2!(other : BaseArray) forall U
    other = other.broadcast_to(shape)
    NumInternal::NDFlatIter2.new(self, other).each do |i, j|
      {% if T == String %}
        i.value = (yield i.value, j.value).to_s
      {% elsif T == Bool %}
        j != 0 ? true : false
      {% else %}
        i.value = T.new(yield i.value, j.value)
      {% end %}
    end
  end

  # Maps a function across two ndarrays.  This returns a copy, so no casting
  # will occur, and the arrays will be broadcast against each other
  # before returning.
  def map2(other : BaseArray(U), &block : T, U -> V) forall U, V
    a, b = NumInternal.broadcast(self, other)
    ret = basetype(V).new(a.shape, 'C')
    NumInternal::NDFlatIter3.new(ret, a, b).each do |i, j, k|
      i.value = yield(j.value, k.value)
    end
    ret
  end

  # Maps a function across three ndarrays in place.  The shapes of the
  # last two arrays must be broadcastable to the original two arrays,
  # so that the operation can be handled in place.  Casting will
  # occur if the result is not of type T
  def map3!(o1 : BaseArray, o2 : BaseArray)
    o1 = NumInternal.bcast_if(o1, shape)
    o2 = NumInternal.bcast_if(o2, shape)
    NumInternal::NDFlatIter3.new(self, o1, o2).each do |i, j, k|
      i.value = T.new(yield i.value, j.value, k.value)
    end
  end

  # Maps a function across three ndarrays.  The arrays will be broadcast
  # against each other and a copy will be made.  The result type is based
  # on the resulting type of the block, and no casting will  occur since
  # a new array is allocated.
  def map3(o1 : BaseArray(U), o2 : BaseArray(V), &block : T, U, V -> W) forall U, V, W
    a, b, c = NumInternal.broadcast(self, o1, o2)
    ret = basetype(W).new(a.shape, 'C')
    NumInternal::NDFlatIter4.new(ret, a, b, c).each do |i, j, k, l|
      i.value = yield(j.value, k.value, l.value)
    end
    ret
  end

  # Duplicates a BaseArray, respecting the passed order of memory
  # provided.  Useful for throwing ndarrays down to LAPACK
  # since they must be in Fortran style order
  def dup(order : Char? = nil)
    contig = uninitialized NumInternal::ArrayFlags
    case order
    when 'C'
      contig = NumInternal::ArrayFlags::Contiguous
    when 'F'
      contig = NumInternal::ArrayFlags::Fortran
    when nil
      contig = @flags & (NumInternal::ArrayFlags::Contiguous | NumInternal::ArrayFlags::Fortran)
    else
      raise NumInternal::ValueError.new(
        "Invalid argument for order.  Valid options or 'C', or 'F'")
    end
    contig_char = contig.contiguous? ? 'C' : 'F'

    ret = self.class.new(@shape, contig_char)
    if (contig & @flags != NumInternal::ArrayFlags::None)
      @buffer.copy_to(ret.buffer, @size)
    else
      ret.iter.zip(iter).each do |i, j|
        i.value = j.value
      end
    end
    ret.update_flags(NumInternal::ArrayFlags::Fortran | NumInternal::ArrayFlags::Contiguous)
    ret
  end

  # Broadcasts an array to a new shape. A readonly view on the original array
  # with the given shape. It is typically not contiguous. Furthermore,
  # more than one element of a broadcasted array may refer to a single
  # memory location.
  def broadcast_to(newshape : Array(Int32))
    NumInternal.broadcast_to(self, newshape)
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
  def as_strided(newshape : Array(Int32), newstrides : Array(Int32))
    NumInternal.as_strided(self, newshape, newstrides)
  end

  # Gives a new shape to an array without changing its data.
  def reshape(newshape : Array(Int32))
    if newshape == shape
      return self
    end
    newsize = 1
    cur_size = size
    autosize = -1
    newshape.each_with_index do |val, i|
      if val < 0
        if autosize >= 0
          raise NumInternal::ValueError.new("Only shape dimension can be automatic")
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
      raise NumInternal::ShapeError.new "Shapes #{shape} cannot be reshaped to #{newshape}"
    end

    stride = uninitialized Int32
    newdims = newshape.size

    if flags & NumInternal::ArrayFlags::Contiguous
      newstrides = NumInternal.shape_to_strides(newshape, 'C')
    else
      newstrides = NumInternal.shape_to_strides(newshape, 'F')
    end

    if flags.fortran? || flags.contiguous?
      newflags = flags.dup
      newflags &= ~NumInternal::ArrayFlags::OwnData
      ret = self.class.new(buffer, newshape, newstrides, newflags)
      ret.update_flags(NumInternal::ArrayFlags::Fortran | NumInternal::ArrayFlags::Contiguous)
      ret
    else
      tmp = dup
      ret = self.class.new(tmp.buffer, newshape, newstrides, tmp.flags.dup)
      ret.update_flags(NumInternal::ArrayFlags::Fortran | NumInternal::ArrayFlags::Contiguous)
      ret
    end
  end

  def reshape(*args)
    reshape(args.to_a)
  end

  # Return a contiguous flattened array.
  # A 1-D array, containing the elements of the input, is returned.
  # A copy is made only if needed.
  def ravel
    reshape(-1)
  end

  # Permute the dimensions of a `Tensor`.  If no order is provided,
  # the dimensions will be reversed, a "true transpose".  Otherwise,
  # the dimensions will be permutated in the order provided.
  def transpose(order : Array(Int32) = [] of Int32)
    newshape = shape.dup
    newstrides = strides.dup
    if order.size == 0
      order = (0...ndims).to_a.reverse
    end
    n = order.size
    if n != ndims
      raise NumInternal::AxisError.new("Axes don't match array")
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
        raise NumInternal::AxisError.new("Invalid axis for this array")
      end
      if reverse_permutation[axis] != -1
        raise NumInternal::AxisError.new("Repeated axis in transpose")
      end
      reverse_permutation[axis] = i
      permutation[i] = axis
    end

    n.times do |i|
      newshape[i] = shape[permutation[i]]
      newstrides[i] = strides[permutation[i]]
    end
    ret = self.class.new(buffer, newshape, newstrides, flags.dup)
    ret.update_flags(NumInternal::ArrayFlags::Contiguous | NumInternal::ArrayFlags::Fortran)
    ret
  end

  def transpose(arr : BaseArray, *args)
    transpose(arr, args.to_a)
  end

  def swapaxes(arr : BaseArray, axis1 : Int32, axis2 : Int32)
    order = (0...arr.ndims).to_a
    order[axis1] = axis2
    order[axis2] = axis1
    transpose(arr, order)
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

    ret = self.class.new(buffer, newshape, newstrides, flags).dup

    1.step(to: shape[axis] - 1) do |_|
      ptr += strides[axis]
      tmp = self.class.new(ptr, newshape, newstrides, flags)
      ret.iter.zip(tmp.iter) do |x, y|
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
      tmp = self.class.new(ptr, newshape, newstrides, flags)
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
        tmp = self.class.new(ptr, newshape, newstrides, ArrayFlags::None)
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

  # Handles an integer indexing argument
  private def normalize_arg(arg : Int32, i : Int32)
    if arg < 0
      arg += shape[i]
    end
    {% if !flag?(:nobounds) %}
      if arg < 0 || arg >= shape[i]
        raise NumInternal::IndexError.new("Index #{arg} out of range for axis #{i} with size #{shape[i]}")
      end
    {% end %}
    return {0, 0, arg}
  end

  # Handles a range indexing argument
  private def normalize_arg(arg : Range, i : Int32)
    start, offset = Indexable.range_to_index_and_count(arg, shape[i])
    {% if !flag?(:nobounds) %}
      if start >= shape[i]
        raise NumInternal::IndexError.new("Index #{start} is out of range for axis #{i} with size #{shape[i]}")
      end
    {% end %}
    return {offset, strides[i], start}
  end

  # Handles a tuple indexing argument
  private def normalize_arg(arg : Tuple(Range(B, E), Int32), i : Int32) forall B, E
    range, step = arg
    abstep = step.abs
    start, offset = Indexable.range_to_index_and_count(range, shape[i])
    {% if flag?(:nobounds) %}
      if start >= shape[i]
        raise NumInternal::IndexError.new("Index #{start} is out of range for axis #{i} with size #{shape[i]}")
      end
    {% end %}
    return {offset // abstep + offset % abstep, step * strides[i], start}
  end

  # Returns a view of an ndarray from a list of indexers.
  # Valid indexers are integers, such as `1` or `2`, ranges, such
  # as `...` or `1...`, and tuples of a range, and an integer representing
  # the step of the index operation, such as `{..., -1}`
  private def slice_internal(args : Array)
    newshape = shape.dup
    newstrides = strides.dup
    newflags = flags.dup
    newflags &= ~NumInternal::ArrayFlags::OwnData

    accessor = args.map_with_index do |arg, i|
      shape_i, strides_i, offset_i = normalize_arg(arg, i)
      newshape[i] = shape_i
      newstrides[i] = strides_i
      offset_i
    end

    i = 0
    newstrides.reject! do |_|
      condition = newshape[i] == 0
      i += 1
      condition
    end
    newshape.reject! { |i| i == 0 }

    ptr = @buffer
    accessor.zip(strides) do |i, j|
      ptr += i * j
    end

    self.class.new(ptr, newshape, newstrides, newflags)
  end

  private def aref_set(*args, value : BaseArray)
    old = self[*args]
    old.map2!(value) { |_, j| T.new(j) }
  end

  private def aref_set(*args, value : U) forall U
    old = self[*args]
    old.map! { |_| T.new(value) }
  end

  # Updates the flags of an array in place.  This by default uses all valid flags
  # as a mask, so if an array must be a view, or read-only, the flags will have
  # to be updated after the fact.
  protected def update_flags(mask : NumInternal::ArrayFlags = NumInternal::ArrayFlags::All)
    if mask & NumInternal::ArrayFlags::Fortran
      if is_fortran_contiguous
        @flags |= NumInternal::ArrayFlags::Fortran

        # mutually exclusive
        if ndims > 1
          @flags &= ~NumInternal::ArrayFlags::Contiguous
        end
      else
        @flags &= ~NumInternal::ArrayFlags::Fortran
      end
    end

    if mask & NumInternal::ArrayFlags::Contiguous
      if is_contiguous
        @flags |= NumInternal::ArrayFlags::Contiguous

        # mutually exclusive
        if ndims > 1
          @flags &= ~NumInternal::ArrayFlags::Fortran
        end
      else
        @flags &= ~NumInternal::ArrayFlags::Contiguous
      end
    end
  end

  # Asserts if an array is fortran contiguous, otherwise known
  # as stored in column major order.  This is not the default
  # layout for arrays, but can provide performance benefits
  # when passing to LaPACK routines since otherwise the
  # array must be transposed in memory.
  private def is_fortran_contiguous
    # Empty arrays are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    ndims.times do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
  end

  # Asserts if an array is c contiguous, otherwise known
  # as stored in row major order.  This is the default memory
  # storage for NDArray
  private def is_contiguous
    # Empty arrays are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional arrays can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    (ndims - 1).step(to: 0, by: -1) do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
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
        raise NumInternal::ShapeError.new("All subarrays must be the same length")
      end
    end

    calc_shape << arr.size
    calc_shape = calculate_shape(arr[0], calc_shape)
    calc_shape
  end
end
