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

require "./internal/broadcast"
require "./internal/constants"
require "./internal/utils"
require "./internal/print"
require "./internal/iteration"
require "./internal/ndindex"
require "../libs/cblas"

# A `Tensor` is a multidimensional container of fixed size, containing
# elements of type T.
#
# The number of dimensions is specified by a `Tensor`'s `shape`, which
# is an `Array` of integers specifying the size of a `Tensor` in each
# dimension.
#
# A `Tensor` can be created from a wide variety of creation methods.
# Including from a scalar value and `shape`, from an `Array`, or from `JSON`.
#
# `Tensor`'s store data using a `Pointer`, and can be sliced/indexed to return
# a `view` into this data.  The slice will share memory with it's parent
# `Tensor`, and modifications to the view will be reflected in the parent
#
# `Tensor`'s cannot be resized, and any operation the changes the total
# number of elements in a `Tensor` will return a new object.
class Tensor(T)
  getter size : Int32
  getter shape : Array(Int32)
  property flags : Num::ArrayFlags
  getter strides : Array(Int32)

  # Creates a new, zero-filled `Tensor` of shape `shape`.  The memory
  # layout of the `Tensor` is specified by `order`.
  #
  # When creating a `Tensor` from a shape alone, the generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   Size of an array along each dimension
  # *order* : Num::OrderType
  #   Internal memory layout of a `Tensor`.  Tensor's can be stored
  #   using either C-style or Fortran-style ordering
  #
  # Examples
  # --------
  # ```
  # Tensor(Int32).new([1, 3]) # => [[0, 0, 0]]
  # ```
  def initialize(shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    @shape = shape.map &.to_i
    @size = @shape.product
    @buffer = Pointer(T).malloc(@size)
    @strides = Num::Internal.shape_to_strides(@shape, order)
    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Creates a new `Tensor` of shape `shape`.  The memory
  # layout of the `Tensor` is specified by `order`.  The `Tensor`
  # is initially populated with the scalar `value`.
  #
  # The generic type of the `Tensor` is inferred from the type
  # of the scalar.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   Size of an array along each dimension
  # *value* : T
  #   Initial value to fill the `Tensor`
  # *order* : Num::OrderType
  #   Internal memory layout of a `Tensor`.  Tensor's can be stored
  #   using either C-style or Fortran-style ordering
  #
  # Examples
  # --------
  # ```
  # Tensor.new([1, 3], 2.4) # => [[2.4, 2.4, 2.4]]
  # ```
  def initialize(
    shape : Array(Int),
    value : T,
    order : Num::OrderType = Num::RowMajor
  )
    @shape = shape.map &.to_i
    @size = @shape.product
    @buffer = Pointer(T).malloc(@size, value)
    @strides = Num::Internal.shape_to_strides(@shape, order)
    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Low level method for creating a `Tensor`. No checks are performed
  # to ensure that the `shape` and `strides` provided are valid for the
  # `Tensor`, so this should primarily be used by lower level methods
  #
  # The generic type of the `Tensor` is inferred from the `buffer`
  #
  # Arguments
  # ---------
  # *buffer* : Pointer(T)
  #   Raw memory for a `Tensor`'s data
  # *shape*  : Array(Int)
  #   Size of an array along each dimension
  # *strides* : Array(Int)
  #   Step along each dimension to reach the next element of a `Tensor`
  #
  # Examples
  # --------
  # ```
  # p = Pointer.malloc(3, 1)
  # shape = [3]
  # strides = [1]
  #
  # Tensor.new(p, shape, strides) # => [1, 1, 1]
  # ```
  def initialize(
    buffer : Pointer(T),
    shape : Array(Int),
    strides : Array(Int)
  )
    @buffer = buffer
    @shape = shape.map &.to_i
    @size = @shape.product
    @strides = strides.map &.to_i
    @flags = Num::ArrayFlags::All
    update_flags
  end

  def initialize(
    buffer : Pointer(T),
    shape : Array(Int),
    strides : Array(Int),
    flags : Num::ArrayFlags
  )
    @buffer = buffer
    @shape = shape.map &.to_i
    @size = @shape.product
    @strides = strides.map &.to_i
    @flags = flags
    update_flags
  end

  # Creates a new `Tensor` with an empty `shape` and `strides`, that contains
  # a scalar value.  This is also similar to what is returned by an
  # indexing operation that slices a `Tensor` along all dimensions.
  #
  # This tensor cannot be indexed or sliced, and all operations on it will
  # return a copy, not a view
  #
  # Arguments
  # ---------
  # *value* : T
  #   A single scalar value for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # Tensor.new(2.5) # => Tensor(2.5)
  # ```
  def initialize(value : T)
    @size = 1
    @buffer = Pointer(T).malloc(@size, value)
    @shape = [] of Int32
    @strides = [] of Int32
    @flags = Num::ArrayFlags::All
  end

  # Creates a new `Tensor` from a provided `shape` and `order`, using values
  # returned from a captured block.  The index passed to the block is one-
  # dimensional.
  #
  # The generic type is inferred from the return type of the block
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   Size of an array along each dimension
  # *order* : Num::OrderType
  #   Internal memory layout of a `Tensor`.  Tensor's can be stored
  #   using either C-style or Fortran-style ordering
  # *block* : Proc(Int32, T)
  #   Proc that takes an integer, and returns a value to be stored
  #   at that flat index of a `Tensor`
  #
  # Examples
  # --------
  # ```
  # Tensor.new([3]) { |i| i } # => [0, 1, 2]
  # ```
  def self.new(
    shape : Array(Int),
    order : Num::OrderType = Num::RowMajor,
    &block : Int32 -> T
  )
    ptr = Pointer(T).malloc(shape.product) do |i|
      yield i
    end
    strides = Num::Internal.shape_to_strides(shape, order)
    new(ptr, shape, strides)
  end

  # Creates a new `Tensor` from a number of rows and columns, as well
  # as a captured block, providing a convenient method to create
  # matrices.
  #
  # The generic type is inferred from the return value of the block
  #
  # Arguments
  # ---------
  # *m* : Int
  #   Number of rows for the `Tensor`
  # *n* : Int
  #   Number of columns for the `Tensor`
  # *block* : Proc(Int32, Int32, T)
  #   Proc that takes an row index and column index, and returns a value to
  #   be stored at that flat index of a `Tensor`
  #
  # Examples
  # --------
  # ```
  # Tensor.new(2, 2) { |i, j| i + j }
  #
  # # [[0, 1],
  # #  [1, 2]]
  # ```
  def self.new(m : Int, n : Int, &block : Int32, Int32 -> T)
    if m == 0 && n == 0
      raise Num::Internal::ShapeError.new("Matrix cannot be empty")
    end
    ptr = Pointer(T).malloc(m * n) do |idx|
      i = idx // n
      j = idx % n
      yield i, j
    end
    shape = [m.to_i, n.to_i]
    strides = Num::Internal.shape_to_strides(shape, Num::RowMajor)
    new(ptr, shape, strides)
  end

  # Creates a `Tensor` from any standard library `Array`.  The shape
  # of the `Array` will be inferred.
  #
  # The generic type will be inferred from the elements of the `Array`
  #
  # Arguments
  # ---------
  # *a* : Array
  #   Array to be turned into a `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Tensor.from_array a # => [1, 2, 3]
  # ```
  def self.from_array(a : Array)
    shape = Num::Internal.calculate_array_shape(a)
    strides = Num::Internal.shape_to_strides(shape)
    ptr = a.flatten.to_unsafe
    new(ptr, shape, strides)
  end

  # Creates a `Tensor` from a JSON object or string.  Only one-dimensional
  # JSON arrays will be accepted, but a provided shape will be used
  # as the output shape of the `Tensor`
  #
  # The generic type must be specified in order to cast the elements from
  # the JSON array
  #
  # Arguments
  # ---------
  # *input* : IO or String
  #   JSON object or String to be converted to a `Tensor`
  # *shape* : Array(Int)
  #   Shape to be used for the output `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = "[1, 2, 3]"
  # Tensor(Int32).from_json(a, [3]) # => [1, 2, 3]
  # ```
  def self.from_json(input, shape : Array(Int))
    t = new(shape)
    iter = t.unsafe_iter
    parser = JSON::PullParser.new(input)
    parser.read_array do
      iter.next.value = T.new(parse)
    end
    t
  end

  # Converts a `Tensor` to a standard library array.  The returned array
  # will always be one-dimensional to avoid return type ambiguity
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.to_a # => [0, 1, 2, 3]
  # ```
  def to_a : Array(T)
    a = Array(T).new(@size)
    each do |e|
      a << e
    end
    a
  end

  # Converts a `Tensor` to a JSON String.  The returned JSON array
  # will always be one-dimensional
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.to_json # => "[0, 1, 2, 3]"
  # ```
  def to_json(b : JSON::Builder) : String
    b.array do
      each &.value.to_json(b)
    end
  end

  # :nodoc:
  def to_s(io : IO)
    io << to_s
  end

  # :nodoc:
  def to_s : String
    Num::Internal.array_to_string(self)
  end

  # :nodoc:
  def gpu : ClTensor(T)
    if @flags.contiguous?
      writer = self
    else
      writer = self.dup(Num::RowMajor)
    end
    c = ClTensor(T).new(@shape)
    Cl.write(Num::ClContext.instance.queue, writer.to_unsafe, c.to_unsafe, UInt64.new(@size * sizeof(T)))
    c
  end

  # :nodoc:
  def inspect(io : IO)
    to_s(io)
  end

  # :nodoc:
  def dtype : T.class
    T
  end

  # :nodoc:
  def value : T
    @buffer.value
  end

  # :nodoc:
  def rank
    @shape.size
  end

  # :nodoc:
  def to_unsafe
    @buffer
  end

  def unique
    u = Set(T).new
    each do |e|
      u << e
    end
    u.to_tensor
  end

  # :nodoc:
  def to_unsafe_c
    {% if T == Complex %}
      @buffer.as(LibCblas::ComplexDouble*)
    {% else %}
      @buffer
    {% end %}
  end

  # :nodoc:
  def to_tensor
    self
  end

  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def [](*args) : Tensor(T)
    slice(args.to_a)
  end

  # :ditto:
  def [](args : Array) : Tensor(T)
    slice(args)
  end

  # The primary method of setting Tensor values.  The slicing behavior
  # for this method is identical to the `[]` method.
  #
  # If a `Tensor` is passed as the value to set, it will be broadcast
  # to the shape of the slice if possible.  If a scalar is passed, it will
  # be tiled across the slice.
  #
  # Arguments
  # ---------
  # *args* : *U
  #   Tuple of arguments.  All but the last argument must be valid
  #   indexer, so a `Range`, `Int`, or `Tuple(Range, Int)`.  The final
  #   argument passed is used to set the values of the `Tensor`.  It can
  #   be either a `Tensor`, or a scalar value.
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[1.., 1..] = 99
  # a
  #
  # # [[ 0,  1],
  # #  [ 2, 99]]
  # ```
  def []=(*args : *U) forall U
    {% begin %}
      set(
        {% for i in 0...U.size - 1 %}
          args[{{i}}],
        {% end %}
        value: args[{{U.size - 1}}]
      )
    {% end %}
  end

  # :ditto:
  def []=(args : Array, value)
    set(args, value)
  end

  # Return a shallow copy of a `Tensor`.  The underlying data buffer
  # is shared, but the `Tensor` owns its other attributes.  Changes
  # to a view of a `Tensor` will be reflected in the original `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).new([3, 3])
  # b = a.view
  # b[...] = 99
  # a
  #
  # # [[99, 99, 99],
  # #  [99, 99, 99],
  # #  [99, 99, 99]]
  # ```
  def view : Tensor(T)
    new_shape = @shape.dup
    new_strides = @strides.dup
    new_flags = @flags.dup
    new_flags &= ~Num::ArrayFlags::OwnData
    self.class.new(@buffer, new_shape, new_strides, new_flags)
  end

  # Return a shallow copy of a `Tensor` with a new dtype.  The underlying
  # data buffer is shared, but the `Tensor` owns its other attributes.
  # The size of the new dtype must be a multiple of the current dtype
  #
  # Arguments
  # ---------
  # *u* : U.class
  #   The data type used to reintepret the underlying data buffer
  #   of a `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.view(Int16) # => [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
  # ```
  def view_as_type(u : U.class) forall U
    s0 = sizeof(T)
    s1 = sizeof(U)
    shape = @shape.dup

    if s0 > s1
      shape[-1] *= (s0 // s1)
    else
      shape[-1] //= (s1 // s0)
    end

    strides = Num::Internal.shape_to_strides(shape)
    buf = @buffer.unsafe_as(Pointer(U))
    Tensor.new(buf, shape, strides)
  end

  # Returns a view of the diagonal of a `Tensor`.  This method only works
  # for two-dimensional arrays.
  #
  # TODO: Implement views for offset diagonals
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(3, 3) { |i, _| i }
  # a.diagonal # => [0, 1, 2]
  # ```
  def diagonal : Tensor(T)
    unless rank == 2
      raise Num::Internal::ShapeError.new("Tensor must be 2D")
    end

    n = @shape.min
    new_shape = [n]
    new_strides = [@strides.sum]
    new_flags = @flags.dup
    t = self.class.new(@buffer, new_shape, new_strides)
    t.flags &= ~Num::ArrayFlags::OwnData
    t
  end

  # Yields the elements of a `Tensor`, always in RowMajor order,
  # as if the `Tensor` was flat.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each do |el|
  #   puts el
  # end
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # ```
  def each
    iter.each do |el|
      yield el.value
    end
  end

  # Yields the memory locations of each element of a `Tensor`, always in
  # RowMajor oder, as if the `Tensor` was flat.
  #
  # This should primarily be used by internal methods.  Methods such
  # as `map!` provided more convenient access to editing the values
  # of a `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_pointer do |el|
  #   puts el.value
  # end
  #
  # # 0
  # # 1
  # # 2
  # # 3
  # ```
  def each_pointer
    iter.each do |el|
      yield el
    end
  end

  # Yields the elements of a `Tensor`, always in RowMajor order,
  # as if the `Tensor` was flat.  Also yields the flat index of each
  # element.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_with_index do |el, i|
  #   puts "#{el}_#{i}"
  # end
  #
  # # 0_0
  # # 1_1
  # # 2_2
  # # 3_3
  # ```
  def each_with_index
    iter.each_with_index do |el, i|
      yield el.value, i
    end
  end

  # Yields the memory locations of each element of a `Tensor`, always in
  # RowMajor oder, as if the `Tensor` was flat.  Also yields the flat
  # index of a `Tensor`
  #
  # This should primarily be used by internal methods.  Methods such
  # as `map!` provided more convenient access to editing the values
  # of a `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new(2, 2) { |i| i }
  # a.each_pointer_with_index do |el, i|
  #   puts "#{el.value}_#{i}"
  # end
  #
  # # 0_0
  # # 1_1
  # # 2_2
  # # 3_3
  # ```
  def each_pointer_with_index
    iter.each_with_index do |el, i|
      yield el, i
    end
  end

  # Yields matrices along the final two dimensions of a `Tensor`.  Useful
  # for methods that operate on matrices in order to map them across
  # a deeply nested `Tensor`.  The matrices yielded are views into
  # the `Tensor`
  #
  # This method only can be called on `Tensor`'s with 3 or more dimensions
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3, 2, 2]) { |i| i }
  # a.each_matrix do |m|
  #   puts m
  # end
  #
  # # [[0, 1],
  # #  [2, 3]]
  # # [[4, 5],
  # #  [6, 7]]
  # # [[ 8,  9],
  # #  [10, 11]]
  # ```
  def each_matrix
    Num::Internal::MatrixIter.new(self).each do |m|
      yield m
    end
  end

  # Yields each view along the axis of a `Tensor`.  Useful for reductions
  # and accumulation operations along an axis.
  #
  # The `Tensor`' yielded is a view, and changes made will be reflected
  # in the parent `Tensor`
  #
  # Arguments
  # ---------
  # *axis* : Int
  #   The axis to view
  # *dims* : Bool
  #   If true, sets the axis shape to `1` for the axis view.  Otherwise
  #   removes the axis when returning views
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.each_axis(1) do |m|
  #   puts m
  # end
  #
  # # [0, 2]
  # # [1, 3]
  #
  # a.each_axis(0, dims: true) do |m|
  #   puts m
  # end
  #
  # # [[0, 1]]
  # # [[2, 3]]
  # ```
  def each_axis(axis : Int = -1, dims : Bool = false)
    Num::Internal::AxisIter.new(self, axis, keepdims: dims).each do |a|
      yield a
    end
  end

  private macro type_inference(lhs, *args)
    value = yield(
      {% for arg in args %}
        {{arg}}.value,
      {% end %}
    )
    {% if T == Bool %}
      {{lhs}}.value = (value ? true : false) && value != 0
    {% elsif T == String %}
      {{lhs}}.value = value.to_s
    {% else %}
      {{lhs}}.value = T.new(value)
    {% end %}
  end

  # Maps a block across a `Tensor`.  The `Tensor` is treated
  # as flat during iteration, and iteration is always done
  # in RowMajor order
  #
  # The generic type of the returned `Tensor` is inferred from
  # the block
  #
  # Arguments
  # ---------
  # *block* Proc(T, U)
  #   Proc to map across the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.map { |e| e + 5 } # => [5, 6, 7]
  # ```
  def map(&block : T -> U) : Tensor(U) forall U
    t = Tensor(U).new(@shape)
    t.iter(self).each do |i, j|
      i.value = yield j.value
    end
    t
  end

  # Maps a block across a `Tensor` in place.  The `Tensor` is treated
  # as flat during iteration, and iteration is always done
  # in RowMajor order
  #
  # Arguments
  # ---------
  # *block* Proc(T, U)
  #   Proc to map across the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # a.map! { |e| e + 5 }
  # a # => [5, 6, 7]
  # ```
  def map!(&block)
    each_pointer do |ptr|
      type_inference ptr, ptr
    end
  end

  # Maps a block across two `Tensors`.  This is more efficient than
  # zipping iterators since it iterates both `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  #
  # The generic type of the returned `Tensor` is inferred from a block
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self`
  # *block* : Proc(T, U, V)
  #   The block to map across both `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  #
  # a.map(b) { |i, j| i + j } # => [0, 2, 4]
  # ```
  def map(t : Tensor(U), &block : T, U -> V) : Tensor(V) forall U, V
    a, b = Num::Internal.broadcast(self, t)
    r = Tensor(V).new(a.shape)
    r.iter(a, b).each do |i, j, k|
      i.value = yield(j.value, k.value)
    end
    r
  end

  # Maps a block across two `Tensors`.  This is more efficient than
  # zipping iterators since it iterates both `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  # The result of the block is stored in `self`.
  #
  # Broadcasting rules still apply, but since this is an in place
  # operation, the other `Tensor` must broadcast to the shape of `self`
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self`
  # *block* : Proc(T, U, T)
  #   The block to map across both `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  #
  # a.map!(b) { |i, j| i + j }
  # a # => [0, 2, 4]
  # ```
  def map!(t : Tensor, &block)
    t = t.as_shape(@shape)
    iter(t).each do |i, j|
      type_inference i, i, j
    end
  end

  # Maps a block across three `Tensors`.  This is more efficient than
  # zipping iterators since it iterates all `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  #
  # The generic type of the returned `Tensor` is inferred from a block
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `v`
  # *v) : Tensor(V)
  #   The third `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `t`
  # *block* : Proc(T, U, V, W)
  #   The block to map across all `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  # c = Tensor.new([3]) { |i| i }
  #
  # a.map(b, c) { |i, j, k| i + j + k } # => [0, 3, 6]
  # ```
  def map(
    t : Tensor(U),
    v : Tensor(V),
    &block : T, U, V -> W
  ) : Tensor(W) forall U, V, W
    a, b, c = Num::Internal.broadcast(self, t, v)
    r = Tensor(W).new(a.shape)
    r.iter(a, b, c).each do |i, j, k, l|
      i.value = yield(j.value, k.value, l.value)
    end
    r
  end

  # Maps a block across three `Tensors`.  This is more efficient than
  # zipping iterators since it iterates all `Tensor`'s in a single
  # call, avoiding overhead from tracking multiple iterators.
  # The result of the block is stored in `self`.
  #
  # Broadcasting rules still apply, but since this is an in place
  # operation, the other `Tensor`'s must broadcast to the shape of `self`
  #
  # Arguments
  # ---------
  # *t* : Tensor(U)
  #   The second `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `v`
  # *v) : Tensor(V)
  #   The third `Tensor` for iteration.  Must be broadcastable
  #   against the `shape` of `self` and `t`
  # *block* : Proc(T, U, V, W)
  #   The block to map across all `Tensor`s
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3]) { |i| i }
  # b = Tensor.new([3]) { |i| i }
  # c = Tensor.new([3]) { |i| i }
  #
  # a.map!(b, c) { |i, j, k| i + j + k }
  # a # => [0, 3, 6]
  # ```
  def map!(t : Tensor, v : Tensor)
    t = t.as_shape(@shape)
    v = v.as_shape(@shape)
    iter(t, v).each do |i, j, k|
      type_inference i, i, j, k
    end
  end

  # Broadcasts a `Tensor` to a new shape.  Returns a read-only
  # view of the original `Tensor`.  Many elements in the `Tensor`
  # will refer to the same memory location, and the result is
  # rarely contiguous.
  #
  # Shapes must be broadcastable, and an error will be raised
  # if broadcasting fails.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   The shape of the desired output `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def as_shape(shape : Array(Int)) : Tensor(T)
    Num::Internal.broadcast_to(self, shape.map &.to_i)
  end

  # Add dimensions to a `Tensor` so that it has at least `n`
  # dimensions.  Zero element `Tensor`s cannot be expanded
  # using this method. If a `Tensor` has more than `n` dimensions
  # it will not be modified
  #
  # Arguments
  # ---------
  # *n*: Int
  #   Minimum number of dimensions for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3].to_tensor
  # a.with_dims(4) # => [[[[1, 2, 3]]]]
  # ```
  #
  def with_dims(n : Int) : Tensor(T)
    if self.rank >= n
      self.view
    else
      d = n - self.rank
      new_shape = [1] * d + @shape
      reshape(new_shape)
    end
  end

  # Expands a `Tensor`s dimensions n times by broadcasting
  # the shape and strides.  No data is copied, and the result
  # is a read-only view of the original `Tensor`
  #
  # Arguments
  # ---------
  # *n* : Int
  #   Number of dimensions to broadcast
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3].to_tensor
  # a.with_broadcast(2)
  #
  # # [[[1]],
  # #
  # #  [[2]],
  # #
  # #  [[3]]]
  # ```
  def with_broadcast(n : Int) : Tensor(T)
    new_shape = @shape + [1] * n
    new_strides = @strides + [0] * n
    as_strided(new_shape, new_strides)
  end

  # `as_strided` creates a view into the `Tensor` given the exact strides
  # and shape. This means it manipulates the internal data structure
  # of a `Tensor` and, if done incorrectly, the array elements can point
  # to invalid memory and can corrupt results or crash your program.
  #
  # It is advisable to always use the original `strides` when
  # calculating new strides to avoid reliance on a contiguous
  # memory layout.
  #
  #
  # Furthermore, `Tensor`s created with this function often contain
  # self overlapping memory, so that two elements are identical.
  # Vectorized write operations on such `Tensor`s will typically be
  # unpredictable. They may even give different results for small,
  # large, or transposed `Tensor`s.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of the new `Tensor`
  # *strides*
  #   Strides of the new `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.as_strided([3, 3], [0, 1])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  def as_strided(shape : Array(Int), strides : Array(Int)) : Tensor(T)
    r = self.class.new(@buffer, shape, strides)
    r.flags &= ~Num::ArrayFlags::OwnData
    r.flags &= ~Num::ArrayFlags::Write
    r
  end

  # Casts a `Tensor` to a new dtype, by making a copy.  Information may
  # be lost when converting between data types, for example Float to Int
  # or Int to Bool.
  #
  # Arguments
  # ---------
  # *u* : U.class
  #   Data type the `Tensor` will be cast to
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.5, 3.5]
  #
  # a.astype(Int32)   # => [1, 2, 3]
  # a.astype(Bool)    # => [true, true, true]
  # a.astype(Float32) # => [1.5, 2.5, 3.5]
  # ```
  def as_type(u : U.class) : Tensor(U) forall U
    r = Tensor(U).new(@shape)
    r.map!(self) do |_, j|
      j
    end
    r
  end

  # :nodoc:
  def real
    {% if T == Complex %}
      self.map &.real
    {% else %}
      self.dup
    {% end %}
  end

  # :nodoc:
  def imag
    {% if T == Complex %}
      self.map &.imag
    {% else %}
      self.dup
    {% end %}
  end

  # Deep-copies a `Tensor`.  If an order is provided, the returned
  # `Tensor`'s memory layout will respect that order.
  #
  # If no order is provided, the `Tensor` will retain it's same
  # memory layout.
  #
  # Arguments
  # ---------
  # *order* : Num::OrderType?
  #   Memory layout to use for the returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.dup # => [1, 2, 3]
  # ```
  def dup(order : Num::OrderType? = nil) : Tensor(T)
    if order.nil?
      if @flags.fortran?
        order = Num::ColMajor
      else
        order = Num::RowMajor
      end
    end
    t = Tensor(T).new(@shape, order)
    t.map!(self) do |_, j|
      j
    end
    t
  end

  # Move a `Tensor` from CPU backed storage to an OpenCL
  # buffer.  This will make a copy of the CPU `Tensor` if
  # it is not C-style contiguous.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3].to_tensor
  # acl = a.opencl
  # ```
  def opencl
    u = @flags.contiguous? ? self : self.dup(Num::RowMajor)
    r = ClTensor(T).new(@shape)
    Cl.write(
      Num::ClContext.instance.queue,
      u.to_unsafe,
      r.to_unsafe,
      UInt64.new(@size * sizeof(T))
    )
    r
  end

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # Arguments
  # ---------
  # *result_shape* : Array(Int)
  #   Result shape for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  def reshape(new_shape : Array(Int))
    newshape = new_shape.map &.to_i
    if newshape == shape
      return self.view
    end
    newsize = 1
    cur_size = size
    autosize = -1
    newshape.each_with_index do |val, i|
      if val < 0
        if autosize >= 0
          raise Num::Internal::ValueError.new("Only shape dimension can be automatic")
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
      raise Num::Internal::ShapeError.new "Shapes #{shape} cannot be reshaped to #{newshape}"
    end

    newstrides = Num::Internal.shape_to_strides(newshape, Num::RowMajor)

    if @flags.contiguous?
      self.class.new(@buffer, newshape, newstrides)
    else
      tmp = self.dup(Num::RowMajor)
      self.class.new(tmp.to_unsafe, newshape, newstrides)
    end
  end

  # :ditto:
  def reshape(*args : Int)
    reshape(args.to_a)
  end

  # Flattens a `Tensor` to a single dimension.  If a view can be created,
  # the reshape operation will not copy data.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.flat # => [0, 1, 2, 3]
  # ```
  def flat : Tensor(T)
    reshape(-1)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # Arguments
  # ---------
  # *axes* : Array(Int)
  #   New ordering of axes for the permuted `Tensor`.  If empty,
  #   a full transpose will occur
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.transpose([2, 0, 1])
  #
  # # [[[ 0,  2,  4],
  # #   [ 6,  8, 10],
  # #   [12, 14, 16],
  # #   [18, 20, 22]],
  # #
  # #  [[ 1,  3,  5],
  # #   [ 7,  9, 11],
  # #   [13, 15, 17],
  # #   [19, 21, 23]]]
  # ```
  def transpose(axes : Array(Int) = [] of Int32)
    order = axes.map &.to_i
    new_shape = @shape.dup
    new_strides = @strides.dup

    if order.size == 0
      order = (0...self.rank).to_a.reverse
    end

    n = order.size
    if n != self.rank
      raise Num::Internal::AxisError.new("Axes do not match Tensor")
    end

    perm = [0] * self.rank
    r_perm = [-1] * self.rank

    n.times do |i|
      axis = order[i]
      if axis < 0
        axis = self.rank + axis
      end
      if axis < 0 || axis >= self.rank
        raise Num::Internal::AxisError.new("Invalid axis for Tensor")
      end
      if r_perm[axis] != -1
        raise Num::Internal::AxisError.new("Repeated axis in transpose")
      end
      r_perm[axis] = i
      perm[i] = axis
    end

    n.times do |i|
      new_shape[i] = shape[perm[i]]
      new_strides[i] = strides[perm[i]]
    end
    t = Tensor(T).new(@buffer, new_shape, new_strides)
    t.flags &= ~Num::ArrayFlags::OwnData
    t
  end

  # :ditto:
  def transpose(*args : Int)
    transpose(args.to_a)
  end

  # Permutes two axes of a `Tensor`.  This will always create a view
  # of the permuted `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Int
  #   First axis of permutation
  # *b* : Int
  #   Second axis of permutation
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.swap_axes(2, 0)
  #
  # # [[[ 0,  6, 12, 18]
  # #   [ 2,  8, 14, 20]
  # #   [ 4, 10, 16, 22]]
  # #
  # #  [[ 1,  7, 13, 19]
  # #   [ 3,  9, 15, 21]
  # #   [ 5, 11, 17, 23]]]
  # ```
  def swap_axes(a : Int, b : Int) : Tensor(T)
    order = (0...self.rank).to_a
    order[a] = b.to_i
    order[b] = a.to_i
    transpose(order)
  end

  private def contiguous(*args : Tensor)
    args.all? do |a|
      a.flags.contiguous?
    end
  end

  # :nodoc:
  def iter
    if contiguous(self)
      Num::Internal::ContigFlatIter1.new(self)
    else
      Num::Internal::NDFlatIter1.new(self)
    end
  end

  # :nodoc:
  def unsafe_iter
    Num::Internal::UnsafeNDFlatIter.new(self)
  end

  def unsafe_axis_iter(axis : Int, dims : Bool = false)
    Num::Internal::UnsafeAxisIter.new(self, axis, dims)
  end

  # :nodoc:
  def iter(a : Tensor)
    if contiguous(self, a)
      Num::Internal::ContigFlatIter2.new(self, a)
    else
      Num::Internal::NDFlatIter2.new(self, a)
    end
  end

  # :nodoc:
  def iter(a : Tensor, b : Tensor)
    if contiguous(self, a, b)
      Num::Internal::ContigFlatIter3.new(self, a, b)
    else
      Num::Internal::NDFlatIter3.new(self, a, b)
    end
  end

  # :nodoc:
  def iter(a : Tensor, b : Tensor, c : Tensor)
    if contiguous(self, a, b, c)
      Num::Internal::ContigFlatIter4.new(self, a, b, c)
    else
      Num::Internal::NDFlatIter4.new(self, a, b, c)
    end
  end

  # :nodoc:
  def map_along_axis(axis : Int)
    if axis < 0
      axis = self.rank + axis
    end

    if axis >= self.rank
      raise Num::Internal::AxisError.new("Axis out of range for Tensor")
    end

    nd = self.rank
    in_dims = (0...nd).to_a
    inarr_view = self.transpose(in_dims[...axis] + in_dims[axis + 1...] + [axis])

    buf = Tensor(T).new(inarr_view.shape)
    buf_permute = (
      in_dims[...axis] +
      in_dims[(nd - 1)...nd] +
      in_dims[axis...(nd - 1)]
    )

    inds = Num::Internal::NDIndex.new(inarr_view.shape[...-1])

    inds.each do |ind|
      buf[ind] = yield inarr_view[ind]
    end
    buf.transpose(buf_permute)
  end

  # :nodoc:
  def yield_along_axis(axis : Int)
    if axis < 0
      axis = self.rank + axis
    end

    if axis >= self.rank
      raise Num::Internal::AxisError.new("Axis out of range for Tensor")
    end

    nd = self.rank
    in_dims = (0...nd).to_a
    inarr_view = self.transpose(in_dims[...axis] + in_dims[axis + 1...] + [axis])

    buf = Tensor(T).new(inarr_view.shape)
    buf_permute = (
      in_dims[...axis] +
      in_dims[(nd - 1)...nd] +
      in_dims[axis...(nd - 1)]
    )

    inds = Num::Internal::NDIndex.new(inarr_view.shape[...-1])

    inds.each do |ind|
      yield inarr_view[ind]
    end
  end

  # :nodoc:
  def reduce_axis(axis : Int, dims : Bool = false)
    if axis < 0
      axis = self.rank + axis
    end

    if axis >= self.rank
      raise Num::Internal::AxisError.new("Axis out of range for Tensor")
    end

    new_shape = @shape.dup
    new_strides = @strides.dup
    ptr = @buffer

    if dims
      new_shape[axis] = 1
      new_strides[axis] = 0
    else
      new_shape.delete_at(axis)
      new_strides.delete_at(axis)
    end

    t = Tensor(T).new(@buffer, new_shape, new_strides, @flags).dup

    1.step(to: @shape[axis] - 1) do
      ptr += @strides[axis]
      t.map!(Tensor(T).new(ptr, new_shape, new_strides)) do |x, y|
        yield x, y
      end
    end
    t
  end

  # :nodoc:
  def accumulate_axis(axis : Int)
    if axis < 0
      axis = self.rank + axis
    end

    if axis >= self.rank
      raise Num::Internal::AxisError.new("Axis out of range for Tensor")
    end

    t = self.dup
    buf = t.to_unsafe
    s0 = @shape.dup
    r0 = @strides.dup
    s0.delete_at(axis)
    r0.delete_at(axis)

    0.step(to: @shape[axis] - 2) do |i|
      f = Tensor(T).new(buf, s0, r0)
      s = Tensor(T).new(buf + @strides[axis], s0, r0)
      s.map!(f) do |i, j|
        yield i, j
      end
      buf += @strides[axis]
    end
    t
  end

  # :nodoc:
  def accumulate
    last = uninitialized T
    ret = Tensor(T).new(@size)
    ret.each_pointer_with_index do |e, i|
      if i == 0
        last = e.value
      else
        e.value = T.new(yield last, e.value)
        last = e.value
      end
    end
    ret
  end

  private def update_flags(m = Num::ArrayFlags::All)
    if m.fortran?
      if is_f_contiguous
        @flags |= Num::ArrayFlags::Fortran

        if self.rank > 1
          @flags &= ~Num::ArrayFlags::Contiguous
        end
      else
        @flags &= ~Num::ArrayFlags::Fortran
      end
    end
    if m.contiguous?
      if is_c_contiguous
        @flags |= Num::ArrayFlags::Contiguous

        if self.rank > 1
          @flags &= ~Num::ArrayFlags::Fortran
        end
      else
        @flags &= ~Num::ArrayFlags::Contiguous
      end
    end
  end

  private def is_f_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end
    s = 1
    self.rank.times do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end

  private def is_c_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    s = 1
    (self.rank - 1).step(to: 0, by: -1) do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end

  private def slice(args : Array)
    new_shape = @shape.dup
    new_strides = @strides.dup
    new_flags = @flags.dup
    new_flags &= ~Num::ArrayFlags::OwnData

    acc = args.map_with_index do |arg, i|
      s_i, st_i, o_i = normalize(arg, i)
      new_shape[i] = s_i
      new_strides[i] = st_i
      o_i
    end

    i = 0
    new_strides.reject! do
      condition = new_shape[i] == 0
      i += 1
      condition
    end

    new_shape.reject! do |j|
      j == 0
    end

    ptr = @buffer

    rank.times do |k|
      if @strides[k] < 0
        ptr += (@shape[k] - 1) * @strides[k].abs
      end
    end

    acc.zip(@strides) do |a, j|
      ptr += a * j
    end

    self.class.new(ptr, new_shape, new_strides, new_flags)
  end

  private def normalize(arg : Int, i : Int32)
    if arg < 0
      arg += @shape[i]
    end
    if arg < 0 || arg >= @shape[i]
      raise Num::Internal::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{@shape[i]}"
      )
    end
    {0, 0, arg.to_i}
  end

  private def normalize(arg : Range, i : Int32)
    s, o = Indexable.range_to_index_and_count(arg, @shape[i])
    if s >= @shape[i]
      raise Num::Internal::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{@shape[i]}"
      )
    end
    {o.to_i, @strides[i], s.to_i}
  end

  private def normalize(arg : Tuple(Range(B, E), Int), i : Int32) forall B, E
    range, step = arg
    abs_step = step.abs
    start, offset = Indexable.range_to_index_and_count(range, @shape[i])
    if start >= @shape[i]
      raise Num::Internal::IndexError.new(
        "Index #{arg} out of range for axis #{i} with size #{@shape[i]}"
      )
    end
    {offset // abs_step + offset % abs_step, step * @strides[i], start}
  end

  private def set(mask : Tensor(Bool), value : Number)
    m = mask.as_shape(@shape)
    map!(m) do |i, c|
      c ? value : i
    end
  end

  # :TODO: Masked iter to iterate through a mask and another
  # `Tensor`
  private def set(mask : Tensor(Bool), value : Tensor)
    m = mask.as_shape(@shape)
    fill = 0
    m.each do |b|
      if b
        fill += 1
      end
    end
    if fill != value.size
      raise Num::Internal::ShapeError.new("Bad value for mask")
    end

    it = value.unsafe_iter
    map!(m) do |i, c|
      c ? it.next.value : i
    end
  end

  private def set(*args, value)
    set(args.to_a, value)
  end

  private def set(args : Array, t : Tensor)
    s = self[args]
    t = t.as_shape(s.shape)
    if t.rank > s.rank
      raise Num::Internal::ShapeError.new("Setting a Tensor with a sequence")
    end
    s.map!(t) do |_, j|
      j
    end
  end

  private def set(args : Array, t : U) forall U
    s = self[args]
    s.map! do
      T.new(t)
    end
  end
end
