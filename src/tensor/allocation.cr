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

class Tensor(T)
  # Initializes a Tensor onto a device with a provided shape and memory
  # layout.
  #
  # ```
  # a = Tensor(Float32).new([3, 3, 2], device: OCL(Float32)) # => GPU Tensor
  # b = Tensor(Float32).new([2, 3, 4])                       # => CPU Tensor
  # ```
  def initialize(shape : Array(Int), order : Num::OrderType = Num::RowMajor, device = CPU(T))
    @storage = device.new(shape, order)
  end

  # Initializes a Tensor onto a device with a provided shape and memory
  # layout, containing a specified value.
  #
  # ```
  # a = Tensor.new([2, 2], 3.5) # => CPU Tensor filled with 3.5
  # ```
  def initialize(shape : Array(Int), value : T, order : Num::OrderType = Num::RowMajor, device = CPU(T))
    @storage = device.new(shape, value, order)
  end

  # Initializes a Tensor onto a device from a provided storage, shape and
  # strides.  This should only be used by internal methods, or if you really
  # believe you know what you're doing.
  #
  # ```
  # c = CPU(Int32).new(10)
  # t = Tensor.new(c, [10], [1]) # => CPU Tensor
  # ```
  def initialize(storage : Num::Backend::Storage(T))
    @storage = storage
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
    storage = Num::Backend.hostptr_to_storage(ptr, shape, order, device)
    new(storage)
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
    storage = Num::Backend.hostptr_to_storage(ptr, [m, n], Num::RowMajor, device)
    new(storage)
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
    storage = Num::Backend.flat_array_to_storage(a.flatten, shape, Num::RowMajor, device)
    new(storage)
  end
end
