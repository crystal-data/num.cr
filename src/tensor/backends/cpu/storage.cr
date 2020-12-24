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

# Data stored using native Crystal pointers, the default
# storage type for Num.cr
struct CPU(T) < Num::Backend::Storage(T)
  getter data : Pointer(T)

  def initialize(data : Pointer(T), shape : Array(Int), strides : Array(Int), size : Int, offset : Int = 0)
    @data = data
    @shape = shape.map &.to_i
    @strides = strides.map &.to_i
    @size = size.to_i
    @offset = offset.to_i
  end

  def initialize(data : Pointer(T), shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    @data = data
    @shape = shape.map &.to_i
    @strides = Num::Internal.shape_to_strides(shape, order)
    @size = shape.product
    @offset = 0
  end

  # Initialize a CPU storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ```
  # Num::CPU.new([2, 3, 4])
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    data = Pointer(T).malloc(shape.product)
    new(data, shape, order)
  end

  # Initialize a CPU storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ```
  # Num::CPU.new([10, 10], 3.4)
  # ```
  def self.new(shape : Array(Int), value : T, order : Num::OrderType = Num::RowMajor)
    strides = Num::Internal.shape_to_strides(shape, order)
    data = Pointer(T).malloc(shape.product, value)
    new(data, shape, strides, shape.product)
  end

  # Allows storage to be passed directly to C libraries
  # and will pass the storage's pointer
  def to_unsafe : Pointer(T)
    @data
  end
end
