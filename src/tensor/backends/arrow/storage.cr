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
struct ARROW(T) < Num::Backend::Storage(T)
  getter data : Arrow::NumericArray

  def initialize(data : Pointer(T), shape : Array(Int), strides : Array(Int), size : Int, offset : Int = 0)
    @data = hostptr_to_arrow(data, shape.product)
    @shape = shape.map &.to_i
    @strides = strides.map &.to_i
    @size = size.to_i
    @offset = offset.to_i
  end

  def initialize(data : Arrow::NumericArray, shape : Array(Int), strides : Array(Int), size : Int, offset : Int = 0)
    @data = data
    @shape = shape.map &.to_i
    @strides = strides.map &.to_i
    @size = size.to_i
    @offset = offset.to_i
  end

  def initialize(data : Pointer(T), shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    @data = hostptr_to_arrow(data, shape.product)
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

  def [](i : Int32) : T
    @data.value(i).unsafe_as(T)
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

  def to_unsafe
    iterator, device = @data.values
    iterator.to_unsafe.unsafe_as(Pointer(T))
  end

  private def hostptr_to_arrow(data : Pointer(T), size : Int32)
    byte_size = size * sizeof(T)
    data = data.unsafe_as(Pointer(UInt8)).to_slice(byte_size)
    buffer = Arrow::Buffer.new data
    cls = Arrow::StringArray
    {% if T == Int8 %}
      cls = Arrow::Int8Array
    {% elsif T == UInt8 %}
      cls = Arrow::UInt8Array
    {% elsif T == Int16 %}
      cls = Arrow::Int16Array
    {% elsif T == UInt16 %}
      cls = Arrow::UInt16Array
    {% elsif T == Int32 %}
      cls = Arrow::Int32Array
    {% elsif T == UInt32 %}
      cls = Arrow::UInt32Array
    {% elsif T == Int64 %}
      cls = Arrow::Int64Array
    {% elsif T == UInt64 %}
      cls = Arrow::UInt64Array
    {% elsif T == Float32 %}
      cls = Arrow::FloatArray
    {% elsif T == Float64 %}
      cls = Arrow::DoubleArray
    {% else %}
      {% raise "Unsupported data type #{T}" %}
    {% end %}
    cls.new size, buffer, nil, 0
  end
end
