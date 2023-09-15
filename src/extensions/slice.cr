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

struct Slice(T)
  # Converts a standard library slice to a `Tensor`.
  # The type of Tensor is inferred from the element type, and alternative
  # shapes can be provided.
  #
  # ## Arguments
  #
  # * device : `Num::Storage` - The storage backend on which to place the `Tensor`
  # * shape : `Array(Int32)?` - An optional shape this slice represents
  #
  # ## Examples
  #
  # ```
  # s = Slice.new(200) { |i| (i + 10).to_u8 }
  # typeof(s.to_tensor) # => Tensor(UInt8, CPU(Float32))
  # ```
  def to_tensor(device = CPU, shape : Array(Int32)? = nil)
    Tensor.from_slice self, device: device, shape: shape
  end
end
