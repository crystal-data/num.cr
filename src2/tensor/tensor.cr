# Copyright (c) 2021 Crystal Data Contributors
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
  include Num::ND(T)

  # Returns the raw Crystal pointer storing a Tensor's data
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.raw => Pointer(UInt8)@0x7f20330c1eb0
  # ```
  getter raw : Pointer(T)

  # Creates an empty Tensor, with a null pointer backing it.
  # Due to how iteration works, an "empty" Tensor has a shape
  # of [], but strides of [1].
  def initialize
    @shape = [] of Int32
    @strides = [1]
    @size = 0
    @offset = 0
    @raw = Pointer(T).null
  end

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
    @raw = Pointer(T).malloc(@size)
    @strides = Num::Internal.shape_to_strides(@shape, order)
    @offset = 0
    self.update_flags
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
    @raw = Pointer(T).malloc(@size, value)
    @strides = Num::Internal.shape_to_strides(@shape, order)
    @offset = 0
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
    raw : Pointer(T),
    shape : Array(Int),
    strides : Array(Int),
    offset : Int
  )
    @raw = raw
    @shape = shape.map &.to_i
    @size = @shape.product
    @strides = strides.map &.to_i
    @offset = offset.to_i
    update_flags
  end

  # :nodoc:
  def initialize(
    raw : Pointer(T),
    shape : Array(Int),
    strides : Array(Int),
    offset : Int,
    flags : Num::ArrayFlags
  )
    @raw = raw
    @shape = shape.map &.to_i
    @size = @shape.product
    @strides = strides.map &.to_i
    @flags = flags
    @offset = offset.to_i
    update_flags
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
    new(ptr, shape, strides, 0)
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
    if m <= 0 || n <= 0
      raise Num::Exceptions::ValueError.new("Matrix must have both rows and columns")
    end
    ptr = Pointer(T).malloc(m * n) do |idx|
      i = idx // n
      j = idx % n
      yield i, j
    end
    shape = [m.to_i, n.to_i]
    strides = Num::Internal.shape_to_strides(shape, Num::RowMajor)
    new(ptr, shape, strides, 0)
  end
end
