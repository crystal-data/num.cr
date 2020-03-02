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

require "./storage"
require "../base_refactor/array"
require "../base_refactor/routines"
require "../base_refactor/exceptions"

class ClTensor(T) < NumInternal::AnyTensor(T)
  getter storage : NumInternal::ClStorage(T)
  getter size : Int32

  def free
    @storage.free
  end

  def to_unsafe
    @storage.to_unsafe
  end

  def basetype(dtype : U.class) forall U
    ClTensor(U)
  end

  def dtype
    T
  end

  def initialize(@shape : Array(Int32))
    @size = @shape.product
    @ndims = @shape.size
    @strides = NumInternal.shape_to_strides(@shape)
    @storage = NumInternal::ClStorage(T).new(@size)
  end

  def initialize(@storage : NumInternal::ClStorage(T), @shape : Array(Int32))
    @size = @shape.product
    @strides = NumInternal.shape_to_strides(@shape)
    @ndims = @shape.size
  end

  def clone
    raise NumInternal::NotImplementedError.new
  end
end
