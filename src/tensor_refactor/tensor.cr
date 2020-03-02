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

require "../base_refactor/array"
require "../base_refactor/constants"
require "../base_refactor/routines"
require "../base_refactor/exceptions"
require "../iter_refactor/flat"
require "../iter_refactor/nd"
require "./storage"
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
    @storage = NumInternal::CpuStorage(T).new(@size, scalar)
    @ndims = 0
    @size = 1
    @shape = [] of Int32
    @strides = [] of Int32
    @flags = Num::ArrayFlags::All
  end

  # Yields an array from a provided shape and a block.  The block only
  # provides the absolute index, not an index dependent on the shape,
  # so if a user wants to handle an arbitrary shape inside the block
  # they need to do that themselves.
  def self.new(shape : Array(Int32), order : Num::ArrayFlags = Num::RowMajor, &block : Int32 -> T)
    size = shape.product
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
    strides = NumInternal.shape_to_strides([nrows, ncols], Num::RowMajor)
    new(data, [nrows, ncols], strides)
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
end

a = AnyArray(Int32).new([3, 3, 2])
puts a[0]
