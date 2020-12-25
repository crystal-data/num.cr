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

abstract struct Num::Backend::Storage(T)
  getter shape : Array(Int32)
  getter strides : Array(Int32)
  getter offset : Int32
  getter size : Int32

  def initialize
    @shape = [] of Int32
    @strides = [1] of Int32
    @offset = 0
    @size = 0
  end

  def rank
    @shape.size
  end

  def is_f_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end
    s = 1
    self.rank.times do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end

  def is_c_contiguous : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    s = 1
    (self.rank - 1).step(to: 0, by: -1) do |i|
      d = @shape[i]
      return true unless d != 0
      return false unless @strides[i] == s
      s *= d
    end
    true
  end
end