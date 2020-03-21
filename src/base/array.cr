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

# The abstract class for any Tensor.  Must have storage, be able
# to free its storage, as well as provide information about
# memory layout and size, and pass itself to low level libraries
abstract class NumInternal::AnyTensor(T)
  getter shape : Array(Int32)
  getter strides : Array(Int32)
  getter ndims : Int32

  def initialize(@shape : Array(Int32), @strides : Array(Int32), @ndims : Int32)
  end

  abstract def storage
  abstract def check_type
  abstract def free
  abstract def clone
  abstract def to_unsafe
  abstract def size
  abstract def dtype
  abstract def basetype(dtype : U.class) forall U

  # Checks that a tensor follows row major memory layout
  def is_c_contiguous : Bool
    # Empty arrays are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional arrays can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    (ndims - 1).step(to: 0, by: -1) do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
  end

  # Checks that a tensor follows col major memory layout
  def is_f_contiguous : Bool
    # Empty arrays are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional arrays can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    ndims.times do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
  end
end
