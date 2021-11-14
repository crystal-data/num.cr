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

class ARROW(T) < Num::Backend::Storage(T)
  private macro allocate_array(data)
    {% if T == Int8 %}
      Arrow::Int8Array.new {{ data.id }}
    {% elsif T == UInt8 %}
      Arrow::UInt8Array.new {{ data.id }}
    {% elsif T == Int16 %}
      Arrow::UInt8Array.new {{ data.id }}
    {% elsif T == UInt16 %}
      Arrow::UInt8Array.new {{ data.id }}
    {% elsif T == Int32 %}
      Arrow::Int32Array.new {{ data.id }}
    {% elsif T == UInt32 %}
      Arrow::UInt32Array.new {{ data.id }}
    {% elsif T == Int64 %}
      Arrow::Int64Array.new {{ data.id }}
    {% elsif T == UInt64 %}
      Arrow::UInt64Array.new {{ data.id }}
    {% elsif T == String %}
      Arrow::StringArray.new {{ data.id }}
    {% else %}
      {% raise "Invalid data type for Apache Arrow backed Tensor" %}
    {% end %}
  end

  private macro allocate_array_from_buffer(*args)
    {% if T == Int8 %}
      Arrow::Int8Array.new {{ *args }}
    {% elsif T == UInt8 %}
      Arrow::UInt8Array.new {{ *args }}
    {% elsif T == Int16 %}
      Arrow::UInt8Array.new {{ *args }}
    {% elsif T == UInt16 %}
      Arrow::UInt8Array.new {{ *args }}
    {% elsif T == Int32 %}
      Arrow::Int32Array.new {{ *args }}
    {% elsif T == UInt32 %}
      Arrow::UInt32Array.new {{ *args }}
    {% elsif T == Int64 %}
      Arrow::Int64Array.new {{ *args }}
    {% elsif T == UInt64 %}
      Arrow::UInt64Array.new {{ *args }}
    {% elsif T == String %}
      Arrow::StringArray.new {{ *args }}
    {% else %}
      {% raise "Invalid data type for Apache Arrow backed Tensor" %}
    {% end %}
  end

  # Initialize an ARROW backed storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of the parent `Tensor`
  # * order : `Array(Int)` - Memory layout of the parent `Tensor`
  #
  # ## Examples
  #
  # ```
  # CPU.new([2, 3, 4])
  # ```
  def initialize(shape : Array(Int), order : Num::OrderType)
    @data = allocate_array Array(T).new(shape.product, T.new(0))
  end

  # Initialize a CPU storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of the parent `Tensor`
  # * strides : `Array(Int)` - Strides of the parent `Tensor`
  #
  # ## Examples
  #
  # ```
  # CPU.new([2, 3, 4])
  # ```
  def initialize(shape : Array(Int), strides : Array(Int))
    @data = allocate_array Array(T).new(shape.product, T.new(0))
  end

  # Initialize a CPU storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of the parent `Tensor`
  # * order : `Array(Int)` - Memory layout of the parent `Tensor`
  # * value : `T` - Initial value to populate the buffer
  #
  # ## Examples
  #
  # ```
  # CPU.new([10, 10], 3.4)
  # ```
  def initialize(shape : Array(Int), order : Num::OrderType, value : T)
    @data = allocate_array Array(T).new(shape.product, value)
  end

  # Initialize a CPU storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of the parent `Tensor`
  # * strides : `Array(Int)` - Strides of the parent `Tensor`
  # * value : `T` - Initial value to populate the buffer
  #
  # ## Examples
  #
  # ```
  # CPU.new([10, 10], 3.4)
  # ```
  def initialize(shape : Array(Int), strides : Array(Int), value : T)
    @data = allocate_array Array(T).new(shape.product, value)
  end

  # Initialize a CPU storage from a hostptr and initial
  # shape.  The shape is not required for this storage type,
  # but is needed by other implementations to ensure copy
  # requirements have the right pointer size.
  #
  # ## Arguments
  #
  # * data : `Pointer(T)` - Existing databuffer for a `Tensor`
  # * shape : `Array(Int)` - Shape of the parent `Tensor`
  # * strides : `Array(Int)` - Strides of the parent `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Pointer(Int32).malloc(10)
  # s = CPU.new(a, [5, 2])
  # ```
  def initialize(data : Pointer(T), shape : Array(Int), strides : Array(Int))
    bytes = Bytes.new(data.unsafe_as(Pointer(UInt8)), shape.product * sizeof(T))
    buffer = Arrow::Buffer.new(bytes)
    @data = allocate_array_from_buffer shape.product, buffer, nil, 0
  end

  # Converts a CPU storage to a crystal pointer
  #
  # ## Examples
  #
  # ```
  # a = CPU(Int32).new([3, 3, 2])
  # a.to_hostptr
  # ```
  def to_hostptr : Pointer(T)
    self.to_unsafe
  end

  # Return a generic class of a specific generic type, to allow
  # for explicit return types in functions that return a different
  # storage type than the parent Tensor
  #
  # ## Examples
  #
  # ```
  # a = CPU(Float32).new([10])
  #
  # # Cannot do
  # # a.class.new ...
  #
  # a.class.base(Float64).new([10])
  # ```
  def self.base(dtype : U.class) : ARROW(U).class forall U
    ARROW(U)
  end

  # :nodoc:
  def update_metadata(shape : Array(Int32), strides : Array(Int32))
  end
end

module Num
  # Deep-copies a `Tensor`.  If an order is provided, the returned
  # `Tensor`'s memory layout will respect that order.
  #
  # If no order is provided, the `Tensor` will retain it's same
  # memory layout.
  #
  # ## Arguments
  #
  # * t : `Tensor(U, CPU(U))` - `Tensor` to duplicate
  # * order : `Num::OrderType` - Memory layout to use for the returned `Tensor`
  #
  # ## Examples
  # -
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.dup # => [1, 2, 3]
  # ```
  def dup(t : Tensor(U, ARROW(U)), order : Num::OrderType = Num::RowMajor) forall U
    result = Tensor(U, ARROW(U)).new(t.shape, order)
    result.map!(t) do |_, j|
      j
    end
    result
  end
end
