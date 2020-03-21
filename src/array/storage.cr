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

struct NumInternal::CpuStorage(T) < NumInternal::StorageBase(T)
  @raw : Pointer(T)
  getter offset : Int32 = 0
  getter size : Int32

  def initialize(@size : Int32)
    @raw = Pointer(T).malloc(@size)
  end

  def initialize(@size : Int32, initial : T)
    @raw = Pointer(T).malloc(@size, initial)
  end

  def initialize(@raw : Pointer(T), @size : Int32, @offset : Int32 = 0)
  end

  def dtype
    T
  end

  def clone
    CpuStorage(T).new(@raw.clone, @size, @offset)
  end

  def to_unsafe
    @raw
  end

  def free
    @raw.free
  end
end
