# Copyright (c) 2021 Crystal Data Contributors
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
  # Converts a Tensor to an Array.  To avoid return
  # type ambiguity this will always return a 1D Array
  #
  # ## Arguments
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [[1, 2], [3, 4]]
  # a.to_a # => [1, 2, 3, 4]
  # ```
  def to_a : Array(T)
    Num.to_a(self)
  end

  # Places a Tensor onto a CPU backend.  No copy is done
  # if the Tensor is already on a CPU
  #
  # ## Arguments
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Float32, OCL(Float32)).ones([3])
  # a.cpu # => [1, 1, 1]
  # ```
  def cpu : Tensor(T, CPU(T))
    Num.cpu(self)
  end

  # Places a Tensor onto an OpenCL backend.  No copy is done
  # if the Tensor is already on a CPU
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Float32, CPU(Float32)).ones([3])
  # a.opencl # => <[3] on OpenCL Backend>
  # ```
  def opencl : Tensor(T, OCL(T))
    Num.opencl(self)
  end

  # Converts a Tensor to a given dtype.  No rounding
  # is done on floating point values.
  #
  # ## Arguments
  #
  # * dtype : `U.class` - desired data type of the returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a.as_type(Int32) # => [1, 2, 3]
  # ```
  def as_type(dtype : U.class) forall U
    Num.as_type(self, dtype)
  end
end
