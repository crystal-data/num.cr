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

require "../spec_helper"

class Foo(T) < Num::BaseArray(T)
  def check_type
    T
  end

  def basetype
    Foo
  end
end

describe Num do
  describe "Convert#astensor" do
    it "leaves tensors alone" do
      t = Tensor(Int32).new([2, 2])
      res = Num.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
      assert_array_equal res, t
    end

    it "coerces other base types to tensors" do
      t = Foo.new([2, 2]) { |i| i }
      res = Num.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end

    it "upscales an array to a tensor" do
      t = [[1, 2, 3], [4, 5, 6]]
      res = Num.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end

    it "upscales a number to a tensor" do
      t = 3
      res = Num.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end
  end
end
