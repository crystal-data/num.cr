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
  getter storage : Num::Backend::Storage(T)

  # Returns the number of dimensions in a Tensor
  #
  # ```
  # a = Tensor(UInt8).new([3, 3, 3, 3])
  # a.rank # => 4
  # ```
  def rank : Int32
    @shape.size
  end

  # Returns the size of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.shape # => [2, 3, 4]
  # ```
  def shape : Array(Int32)
    @storage.shape
  end

  # Returns the step of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([3, 3, 2])
  # a.shape # => [4, 2, 1]
  # ```
  def strides : Array(Int32)
    @storage.strides
  end

  # Returns the offset of a Tensor's data
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.offset # => 0
  # ```
  def offset : Int32
    @storage.offset
  end

  # Returns the size of a Tensor along each dimension
  #
  # ```
  # a = Tensor(UInt8).new([2, 3, 4])
  # a.shape # => [2, 3, 4]
  # ```
  def size : Int32
    @storage.size
  end
end
