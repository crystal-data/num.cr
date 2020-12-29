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

struct OCL(T) < Num::Backend::Storage(T)
  getter data : LibCL::ClMem

  # Initialize an OpenCL storage from an initial capacity.
  # The data will be filled with zeros
  #
  # ```
  # OCL.new([100])
  # ```
  def initialize(shape : Array(Int))
    @data = Cl.buffer(Num::ClContext.instance.context, shape.product.to_u64, dtype: T)
  end

  # Initialize an OpenCL storage from an initial capacity and
  # an initial value, which will fill the buffer
  #
  # ```
  # OCL.new([10, 10], 3.4)
  # ```
  def initialize(shape : Array(Int), value : T)
    @data = Cl.buffer(Num::ClContext.instance.context, shape.product.to_u64, dtype: T)
    Cl.fill(Num::ClContext.instance.queue, data, value, shape.product.to_u64)
  end

  # Initialize an OpenCL storage from a standard library Crystal
  # pointer
  #
  # ```
  # ptr = Pointer(Int32).malloc(9)
  # OCL.new(ptr, [3, 3])
  # ```
  def self.from_hostptr(hostptr : Pointer(T), shape : Array(Int))
    storage = OCL(T).new(shape)
    Cl.write(Num::ClContext.instance.queue, hostptr, storage.data, (shape.product * sizeof(T)).to_u64)
    storage
  end
end
