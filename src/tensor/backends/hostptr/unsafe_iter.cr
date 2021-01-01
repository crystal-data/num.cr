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

struct Num::Internal::UnsafeNDFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @track : Pointer(Int32)
  @dim : Int32

  def initialize(arr : Tensor(T, S))
    @ptr = arr.data.to_hostptr
    @shape = arr.shape.to_unsafe
    @strides = arr.strides.to_unsafe
    @track = Pointer(Int32).malloc(arr.rank, 0)
    @dim = arr.rank - 1
    arr.rank.times do |i|
      if @strides[i] < 0
        @ptr += (@shape[i] - 1) * @strides[i].abs
      end
    end
  end

  def next
    ret = @ptr
    @dim.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]
      if @track[i] == shape_i
        @track[i] = 0
        @ptr -= (shape_i - 1) * stride_i
        next
      end
      @ptr += stride_i
      break
    end
    ret
  end
end
