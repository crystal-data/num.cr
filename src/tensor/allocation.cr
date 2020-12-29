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

class Tensor(T, S)
  private def initialize(@data : S, shape : Array(Int), order : Num::OrderType = Num::RowMajor, dtype : T.class = T)
    assert_types
    @shape = shape.map &.to_i
    @strides = Num::Internal.shape_to_strides(shape, order)
    @size = @shape.product
    @offset = 0
  end

  private def initialize(@data : S, shape : Array(Int), from_array : Array(T))
    assert_types
    @shape = shape.map &.to_i
    @strides = Num::Internal.shape_to_strides(shape, Num::RowMajor)
    @size = @shape.product
    @offset = 0
  end

  # Initializes a Tensor onto a device with a provided shape and memory
  # layout.
  #
  # ```
  # a = Tensor(Float32).new([3, 3, 2], device: OCL(Float32)) # => GPU Tensor
  # b = Tensor(Float32).new([2, 3, 4])                       # => CPU Tensor
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    data = S.new(shape)
    new(data, shape, order, T)
  end

  # Initializes a Tensor onto a device with a provided shape and memory
  # layout, containing a specified value.
  #
  # ```
  # a = Tensor.new([2, 2], 3.5) # => CPU Tensor filled with 3.5
  # ```
  def self.new(shape : Array(Int), value : T, device = CPU(T), order : Num::OrderType = Num::RowMajor)
    data = device.new(shape, value)
    new(data, shape, order, T)
  end

  # Creates a Tensor from a block onto a specified device. The type of the
  # Tensor is inferred from the return type of the block
  #
  # ```
  # a = Tensor.new([3, 3, 2]) { |i| i } # => Int32 Tensor stored on a CPU
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor, device = CPU, &block : Int32 -> T)
    ptr = Pointer.malloc(shape.product) do |index|
      yield index
    end
    storage = device.from_hostptr(ptr, shape)
    new(storage, shape, order, T)
  end

  # Creates a matrix Tensor from a block onto a specified device.  The type
  # of the Tensor is inferred from the return type of the block
  #
  # ```
  # a = Tensor.new(3, 3) { |i, j| i / j } # => Float64 Tensor stored on a CPU
  # ```
  def self.new(m : Int, n : Int, device = CPU, &block : Int32, Int32 -> T)
    ptr = Pointer.malloc(m * n) do |idx|
      i = idx // n
      j = idx % n
      yield i, j
    end
    storage = device.from_hostptr(ptr, [m, n])
    new(storage, [m, n], Num::RowMajor, T)
  end

  # Creates a Tensor from a standard library array onto a specified device.
  # The type of Tensor is inferred from the innermost element type, and
  # the Array's shape must be uniform along all subarrays.
  #
  # ```
  # a = [[1, 2], [3, 4], [5, 6]]
  # Tensor.from_array(a, device: OCL) # => [3, 2] Tensor stored on a GPU
  # ```
  def self.from_array(a : Array, device = CPU)
    shape = Num::Internal.recursive_array_shape(a)
    flat = a.flatten
    storage = device.from_hostptr(flat.to_unsafe, shape)
    new(storage, shape, from_array: flat)
  end
end
