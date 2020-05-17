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

require "../base/storage"
require "../base/exceptions"
require "./global"
require "opencl"

struct NumInternal::ClStorage(T) < NumInternal::StorageBase(T)
  @raw : LibCL::ClMem
  getter size : Int32
  getter offset : Int32

  def initialize(@size : Int32, @offset : Int32 = 0)
    @raw = Cl.buffer(Num::ClContext.instance.context, UInt64.new(@size), dtype: T)
  end

  def initialize(@size : Int32, value : T, @offset : Int32 = 0)
    @raw = Cl.buffer(Num::ClContext.instance.context, UInt64.new(@size), dtype: T)
    Cl.fill(Num::ClContext.instance.queue, @raw, value, UInt64.new(size))
  end

  def dtype
    T
  end

  def free
    Cl.release_buffer(@raw)
  end

  def to_unsafe
    @raw
  end
end
