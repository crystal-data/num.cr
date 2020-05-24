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

describe AnyArray do
  describe "BaseArray#indexing" do
    it "accesses scalar value" do
      m = AnyArray.new([3, 2, 2]) { |i| i }
      m[0, 0, 1].value.should eq 1
    end

    it "identifies scalar value out of range" do
      m = AnyArray.new([4]) { |i| i }
      expect_raises(NumInternal::IndexError) do
        m[6]
      end
    end

    it "valid conversion of negative indices" do
      m = AnyArray.new([3]) { |i| i }
      m[-1].value.should eq 2
    end

    it "set a scalar value" do
      m = AnyArray.new([2, 2]) { |i| i }
      m[0, 0] = 100
      expected = AnyArray.from_array [[100, 1], [2, 3]]
      assert_array_equal m, expected
    end

    it "set a scalar value of range raises" do
      m = AnyArray.new([3]) { |i| i }
      expect_raises(NumInternal::IndexError) do
        m[10] = 5
      end
    end

    it "sets based on negative indices" do
      m = AnyArray.new([3]) { |i| i }
      m[-1] = 100
      expected = AnyArray.from_array [0, 1, 100]
      assert_array_equal expected, m
    end

    it "accesses a slice of an nd array" do
      m = AnyArray.new([3, 2, 2]) { |i| i }
      slice = m[0]
      expected = AnyArray.from_array [[0, 1], [2, 3]]
      assert_array_equal slice, expected
    end

    it "accesses strided slice of an nd array" do
      m = AnyArray.new([3, 2, 2]) { |i| i }
      slice = m[..., 0]
      expected = AnyArray.from_array [[0, 1], [4, 5], [8, 9]]
      assert_array_equal slice, expected
    end

    it "cannot assign a sequence to an array" do
      m = AnyArray.new([2, 4]) { |i| i }
      n = AnyArray.new([1, 4]) { |i| i }
      expect_raises(NumInternal::ShapeError) do
        m[1] = n
      end
    end
  end
end
