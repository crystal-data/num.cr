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
require "../tensor/tensor"

abstract class Frame::Index(T)
  abstract def data
  abstract def engine
  abstract def shape
  abstract def dtype
  abstract def size
  abstract def to_num
end

class Frame::IntegerIndex(Int32) < Frame::Index(Int32)
  getter data : Tensor(Int32)
  getter engine : Hash(Int32, Int32)

  def initialize(size : Int32)
    @data = Tensor(Int32).range(size)
    @engine = Hash(Int32, Int32).new
    @data.each_with_index do |e, i|
      if @engine.has_key?(e)
        raise "Index must be unique"
      end
      @engine[e] = i
    end
  end

  delegate shape, dtype, size, to: @data

  def to_num
    @data
  end

  def to_s(io)
    io << "#{self.class}"
    @data.to_s(io)
  end
end
