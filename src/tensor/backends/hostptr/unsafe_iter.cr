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

# :nodoc:
struct Num::Internal::UnsafeNDFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @track : Pointer(Int32)
  @dim : Int32

  def initialize(arr : Tensor(T, U)) forall U
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

# :nodoc:
struct Num::Internal::UnsafeAxisIter(T, S)
  include Iterator(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @inc : Int32
  @offset : Int32
  @tmp : Tensor(T, S)
  @total : Int32
  @yielded : Int32 = 0
  @axis : Int32

  def initialize(arr : Tensor(T, S), @axis : Int32 = -1, keepdims = false) forall U
    if @axis < 0
      @axis += arr.rank
    end
    unless @axis < arr.rank
      raise "Axis out of range for array"
    end

    @shape = arr.shape.dup
    @strides = arr.strides.dup
    @inc = arr.strides[axis]
    @offset = arr.offset

    if keepdims
      @shape[axis] = 1
      @strides[axis] = 0
    else
      @shape.delete_at(axis)
      @strides.delete_at(axis)
    end

    @tmp = arr.class.new(arr.data, @shape, @strides, @offset, T)

    @total = arr.shape[axis]
  end

  def next
    ret = @tmp
    @offset += @inc
    @tmp = Tensor.new(@tmp.data, @shape, @strides, @offset, T)
    ret
  end
end
