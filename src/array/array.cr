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

require "../base/array"
require "../base/constants"
require "../base/routines"
require "../base/exceptions"
require "../iter/flat"
require "../iter/nd"
require "../iter/macro"
require "../iter/axes"
require "./storage"
require "./broadcast"
require "./print"
require "complex"

class AnyArray(T) < NumInternal::AnyTensor(T)
  # Stores an arrays data buffer
  getter storage : NumInternal::CpuStorage(T)

  # Flags indicating the memory layout of an array
  getter flags : Num::ArrayFlags

  # Number of elements in an array
  getter size : Int32

  # This should be overwritten by all base classes.
  # Determines if an array is initialized containing
  # a valid generic type
  def check_type
  end

  # Frees the memory of an array
  def free
    @storage.free
  end

  # Clones an array
  def clone
    raise NumInternal::NotImplementedError.new
  end

  # Returns the raw memory information of an array
  def to_unsafe
    @storage.to_unsafe
  end

  # Returns the generic type of an array
  def dtype
    T
  end

  # Returns the basetype of an array with the ability to
  # override the generic type
  def basetype(dtype : U.class) forall U
    AnyArray(U)
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

  # Checks the write flag on an array, raising a WriteError
  # if the array is read-only, such as an array returned
  # from a method like as_strided
  def write?
    if !flags.write?
      raise NumInternal::WriteError.new("Array is read only")
    end
  end

  # Initializes an ndarray from a provided shape and memory layout.
  # Flags will be auto-computed
  def initialize(@shape : Array(Int32), order : Num::OrderType = Num::RowMajor)
    check_type
    @size = @shape.product
    @ndims = @shape.size
    @storage = NumInternal::CpuStorage(T).new(@size)
    @strides = NumInternal.shape_to_strides(@shape, order)

    if @shape == [] of Int32
      @shape = [0]
      @strides = [1]
    end

    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Initialization method for a generic buffer, shape and strides.
  # This method updates flags on the passed array, if the array
  # is read only, its flags need to be updated later
  def initialize(buffer : Pointer(T), @shape, @strides, @flags = Num::ArrayFlags::All)
    check_type
    @size = @shape.product
    @storage = NumInternal::CpuStorage(T).new(buffer, @size)
    @ndims = @shape.size
    update_flags
  end

  # Crates a scalar tensor, that acts like a scalar while still being
  # a Tensor.  This was primarily added so that indexing operations
  # could return single elements without having a union return type.
  def initialize(scalar : T)
    @size = 1
    @storage = NumInternal::CpuStorage(T).new(@size, scalar)
    @ndims = 0
    @shape = [] of Int32
    @strides = [] of Int32
    @flags = Num::ArrayFlags::All
  end

  # Yields an array from a provided shape and a block.  The block only
  # provides the absolute index, not an index dependent on the shape,
  # so if a user wants to handle an arbitrary shape inside the block
  # they need to do that themselves.
  def self.new(shape : Array(Int32), order : Num::OrderType = Num::RowMajor, &block : Int32 -> T)
    size = shape.product
    ptr = Pointer(T).malloc(size) do |i|
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
    strides = NumInternal.shape_to_strides([nrows, ncols], Num::RowMajor)
    new(data, [nrows, ncols], strides)
  end

  def self.from_array(array : Array)
    newshape = calculate_shape(array)
    newstrides = NumInternal.shape_to_strides(newshape)
    ptr = array.flatten.to_unsafe
    new(ptr, newshape, newstrides)
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

  def value
    to_unsafe.value
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
  def []=(idx : Array, assign : AnyArray(T))
    write?
    old = self[idx]
    # ameba:disable Lint/UnusedArgument
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

  # Shallow copies the array.  Shape and strides are copied, but
  # the underlying data is not.  The returned array does
  # not own its own data, and its base reflects that.
  def dup_view
    newshape = @shape.dup
    newstrides = @strides.dup
    newflags = @flags.dup
    newflags &= ~Num::ArrayFlags::OwnData
    self.class.new(to_unsafe, newshape, newstrides, newflags)
  end

  # Returns a view of the diagonal of an array.  Only valid if
  # the array has two dimensions.  Offsets are not supported.
  def diag_view
    raise NumInternal::ShapeError.new("Array must be two-dimensional") unless ndims == 2
    nel = @shape.min
    newshape = [nel]
    newstrides = [@strides.sum]
    newflags = @flags.dup
    newflags &= ~Num::ArrayFlags::OwnData
    ret = self.class.new(to_unsafe, newshape, newstrides, newflags)
    ret.update_flags(Num::ArrayFlags::Fortran | Num::ArrayFlags::Contiguous)
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

  def iter2(other : AnyArray(U)) forall U
    if flags.contiguous? && other.flags.contiguous?
      NumInternal::ContigFlatIter2.new(self, other)
    else
      NumInternal::NDFlatIter2.new(self, other)
    end
  end

  def iter3(o1 : AnyArray(U), o2 : AnyArray(V)) forall U, V
    if flags.contiguous? && o1.flags.contiguous? && o2.flags.contiguous?
      NumInternal::ContigFlatIter3.new(self, o1, o2)
    else
      NumInternal::NDFlatIter3.new(self, o1, o2)
    end
  end

  def axis(axis = -1, keepdims = false)
    NumInternal::AxisIter.new(self, axis, keepdims)
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
    ret = basetype(U).new(shape)
    ret.iter2(self).each do |i, j|
      i.value = yield j.value
    end
    ret
  end

  # Maps a function across two ndarrays.  This is an inplace operation, so the
  # shape of the other array must be broadcastable to the shape of the input
  # array.
  def map2!(other : AnyArray) forall U
    other = NumInternal.broadcast_to(other, shape)
    # ameba:disable Lint/UnusedArgument
    self.iter2(other).each do |i, j|
      {% if T == String %}
        i.value = (yield i.value, j.value).to_s
      {% elsif T == Bool %}
        i.value = (yield i.value, j.value) ? true : false
      {% else %}
        i.value = T.new(yield i.value, j.value)
      {% end %}
    end
  end

  # Maps a function across two ndarrays.  This returns a copy, so no casting
  # will occur, and the arrays will be broadcast against each other
  # before returning.
  def map2(other : AnyArray(U), &block : T, U -> V) forall U, V
    a, b = NumInternal.broadcast2(self, other)
    ret = basetype(V).new(a.shape)
    ret.iter3(a, b).each do |i, j, k|
      i.value = yield(j.value, k.value)
    end
    ret
  end

  # Maps a function across three ndarrays in place.  The shapes of the
  # last two arrays must be broadcastable to the original two arrays,
  # so that the operation can be handled in place.  Casting will
  # occur if the result is not of type T
  def map3!(o1 : AnyArray, o2 : AnyArray)
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
  def map3(o1 : AnyArray(U), o2 : AnyArray(V), &block : T, U, V -> W) forall U, V, W
    a, b, c = NumInternal.broadcast(self, o1, o2)
    ret = basetype(W).new(a.shape)
    NumInternal::NDFlatIter4.new(ret, a, b, c).each do |i, j, k, l|
      i.value = yield(j.value, k.value, l.value)
    end
    ret
  end

  def broadcast_to(shape : Array(Int32))
    NumInternal.broadcast_to(self, shape)
  end

  def as_strided(shape : Array(Int32), strides : Array(Int32))
    NumInternal.as_strided(self, shape, strides)
  end

  def astype(dtype : U.class) forall U
    ret = self.basetype(U).new(@shape)
    {% if U == Bool %}
      ret.map2!(self) do |_, j|
        j != 0
      end
    {% else %}
      ret.map2!(self) do |_, j|
        U.new(j.value)
      end
    {% end %}
    ret
  end

  # Duplicates a BaseArray, respecting the passed order of memory
  # provided.  Useful for throwing ndarrays down to LAPACK
  # since they must be in Fortran style order
  def dup(order : Num::OrderType? = nil)
    contig = uninitialized Num::ArrayFlags
    case order
    when Num::RowMajor
      contig = Num::ArrayFlags::Contiguous
    when Num::ColMajor
      contig = Num::ArrayFlags::Fortran
    when nil
      contig = @flags & (Num::ArrayFlags::Contiguous | Num::ArrayFlags::Fortran)
    else
      raise NumInternal::ValueError.new(
        "Invalid argument for order.  Valid options are RowMajor, or ColMajor")
    end
    contig_char = contig.fortran? ? Num::ColMajor : Num::RowMajor

    ret = self.class.new(@shape, contig_char)
    if (contig & @flags != Num::ArrayFlags::None)
      self.to_unsafe.copy_to(ret.to_unsafe, @size)
    else
      ret.map2!(self) do |_, j|
        j
      end
    end
    ret.update_flags
    ret
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

    if flags & Num::ArrayFlags::Contiguous
      newstrides = NumInternal.shape_to_strides(newshape, Num::RowMajor)
    else
      newstrides = NumInternal.shape_to_strides(newshape, Num::ColMajor)
    end

    if flags.fortran? || flags.contiguous?
      newflags = flags.dup
      newflags &= ~Num::ArrayFlags::OwnData
      ret = self.class.new(to_unsafe, newshape, newstrides, newflags)
      ret.update_flags(Num::ArrayFlags::Fortran | Num::ArrayFlags::Contiguous)
      ret
    else
      tmp = dup
      ret = self.class.new(tmp.to_unsafe, newshape, newstrides, tmp.flags.dup)
      ret.update_flags(Num::ArrayFlags::Fortran | Num::ArrayFlags::Contiguous)
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
    ret = self.class.new(to_unsafe, newshape, newstrides, flags.dup)
    ret.update_flags(Num::ArrayFlags::Contiguous | Num::ArrayFlags::Fortran)
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

  # Reduces a tensor along an axis
  def reduce_fast(axis, keepdims = false)
    if axis < 0
      axis = ndims + axis
    end
    raise "Axis out of range for this array" unless axis < ndims
    newshape = shape.dup
    newstrides = strides.dup
    ptr = to_unsafe

    if !keepdims
      newshape.delete_at(axis)
      newstrides.delete_at(axis)
    else
      newshape[axis] = 1
      newstrides[axis] = 0
    end

    ret = self.class.new(to_unsafe, newshape, newstrides, flags).dup

    1.step(to: shape[axis] - 1) do |_|
      ptr += strides[axis]
      tmp = self.class.new(ptr, newshape, newstrides, flags)
      ret.map2!(tmp) do |x, y|
        yield x, y
      end
    end
    ret
  end

  # Updates the flags of an array in place.  This by default uses all valid flags
  # as a mask, so if an array must be a view, or read-only, the flags will have
  # to be updated after the fact.
  protected def update_flags(mask : Num::ArrayFlags = Num::ArrayFlags::All)
    if mask & Num::ArrayFlags::Fortran
      if is_f_contiguous
        @flags |= Num::ArrayFlags::Fortran

        # mutually exclusive
        if ndims > 1
          @flags &= ~Num::ArrayFlags::Contiguous
        end
      else
        @flags &= ~Num::ArrayFlags::Fortran
      end
    end

    if mask & Num::ArrayFlags::Contiguous
      if is_c_contiguous
        @flags |= Num::ArrayFlags::Contiguous

        # mutually exclusive
        if ndims > 1
          @flags &= ~Num::ArrayFlags::Fortran
        end
      else
        @flags &= ~Num::ArrayFlags::Contiguous
      end
    end
  end

  # Returns a view of an ndarray from a list of indexers.
  # Valid indexers are integers, such as `1` or `2`, ranges, such
  # as `...` or `1...`, and tuples of a range, and an integer representing
  # the step of the index operation, such as `{..., -1}`
  private def slice_internal(args : Array)
    newshape = shape.dup
    newstrides = strides.dup
    newflags = flags.dup
    newflags &= ~Num::ArrayFlags::OwnData

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
    newshape.reject! { |j| j == 0 }

    ptr = to_unsafe

    @ndims.times do |k|
      if @strides[k] < 0
        ptr += (@shape[k] - 1) * @strides[k].abs
      end
    end

    accessor.zip(strides) do |a, j|
      ptr += a * j
    end

    self.class.new(ptr, newshape, newstrides, newflags)
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

  # Sets an array's values based on another array.
  private def aref_set(*args, value : AnyArray(T))
    old = self[*args]
    old.map2!(value) { |_, j| T.new(j) }
  end

  # Sets an arrays values based on a scalar or other type,
  # if the other type can be cast to the type of the array
  private def aref_set(*args, value : U) forall U
    old = self[*args]
    old.map! { |_| T.new(value) }
  end
end
