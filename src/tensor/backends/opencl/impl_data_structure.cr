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

class OCL(T) < Num::Backend::Storage(T)
  # Data buffer containing the data associated with the parent `Tensor`
  getter data : LibCL::ClMem

  # Data buffer containing the shape associated with the parent `Tensor`
  getter shape : LibCL::ClMem

  # Data buffer containing the strides associated with the parent `Tensor`
  getter strides : LibCL::ClMem

  # Total size of the underlying data buffer.  Keeps track of the total
  # size of a buffer if a `Tensor` has been sliced
  getter total_size : Int32

  # Returns the underlying OpenCL memory buffer associated with a `Tensor`
  def to_unsafe : LibCL::ClMem
    @data
  end
end

module Num
  # :nodoc:
  def tensor_to_string(arr : Tensor(U, OCL(U))) forall U
    "<#{arr.shape} on OpenCL Backend>"
  end
end
