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

# Data stored using an OpenCL Memory buffer, either on a GPU or a CPU
struct OCL(T) < Num::Backend::Storage(T)
  getter data : LibCL::ClMem

  def initialize(data : LibCL::ClMem, shape : Array(Int), strides : Array(Int), size : Int, offset : Int = 0)
    @data = data
    @shape = shape.map &.to_i
    @strides = strides.map &.to_i
    @size = size.to_i
    @offset = offset.to_i
  end

  # Initialize an OpenCL storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ```
  # Num::OCL.new([100])
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    strides = Num::Internal.shape_to_strides(shape, order)
    data = Cl.buffer(Num::ClContext.instance.context, shape.product.to_u64, dtype: T)
    new(data, shape, strides, shape.product)
  end

  # Initialize an OpenCL storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ```
  # Num::OCL.new([10, 10], 3.4)
  # ```
  def self.new(shape : Array(Int), value : T, order : Num::OrderType = Num::RowMajor)
    strides = Num::Internal.shape_to_strides(shape, order)
    data = Cl.buffer(Num::ClContext.instance.context, shape.product.to_u64, dtype: T)
    Cl.fill(Num::ClContext.instance.queue, data, value, shape.product.to_u64)
    new(data, shape, strides, shape.product)
  end
end
