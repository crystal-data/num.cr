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

struct CPU(T) < Num::Backend::Storage(T)
  # Initialize a CPU storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ```
  # CPU.new([2, 3, 4])
  # ```
  def initialize(shape : Array(Int))
    @data = Pointer(T).malloc(shape.product)
  end

  # Initialize a CPU storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ```
  # CPU.new([10, 10], 3.4)
  # ```
  def initialize(shape : Array(Int), value : T)
    @data = Pointer(T).malloc(shape.product, value)
  end

  # Initialize a CPU storage from a hostptr and initial
  # shape.  The shape is not required for this storage type,
  # but is needed by other implementations to ensure copy
  # requirements have the right pointer size.
  #
  # ```
  # a = Pointer(Int32).malloc(10)
  # s = CPU.new(a, [5, 2])
  # ```
  def initialize(data : Pointer(T), shape : Array(Int))
    @data = data
  end

  # Converts a CPU storage to a crystal pointer
  #
  # ```
  # a = CPU(Int32).new([3, 3, 2])
  # a.to_hostptr
  # ```
  def to_hostptr : Pointer(T)
    @data
  end

  # Return a generic class of a specific generic type, to allow
  # for explicit return types in functions that return a different
  # storage type than the parent Tensor
  #
  # ```
  # a = CPU(Float32).new([10])
  #
  # # Cannot do
  # # a.class.new ...
  #
  # a.class.base(Float64).new([10])
  # ```
  def self.base(dtype : U.class) : CPU(U).class forall U
    CPU(U)
  end
end

module Num
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
  def dup(t : Tensor(U, CPU(U)), order : Num::OrderType = Num::RowMajor) forall U
    result = Tensor(U, CPU(U)).new(t.shape, order)
    result.map!(t) do |_, j|
      j
    end
    result
  end
end
