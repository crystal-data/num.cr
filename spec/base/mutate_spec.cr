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
  describe "BaseArray#reshape" do
    it "reshapes a tensor to 2d" do
      a = AnyArray.from_array [1, 2, 3, 4]
      expected = AnyArray.from_array [[1, 2], [3, 4]]
      res = a.reshape(2, 2)
      assert_array_equal res, expected
    end

    it "infer reshape dimension" do
      a = AnyArray.from_array [1, 2, 3, 4]
      expected = AnyArray.from_array [[1, 2], [3, 4]]
      res = a.reshape(-1, 2)
      assert_array_equal res, expected
    end

    it "raise on bad reshape" do
      a = AnyArray.new([10]) { |i| i }
      expect_raises(NumInternal::ShapeError) do
        a.reshape(3, 3)
      end
    end

    it "raises on infer two dimensions" do
      a = AnyArray.new([12]) { |i| i }
      expect_raises(NumInternal::ValueError) do
        a.reshape(2, -1, -1)
      end
    end
  end

  describe "BaseArray#ravel" do
    it "flattens array" do
      desired = AnyArray.new([12]) { |i| i }
      result = desired.reshape(3, 2, 2).ravel
      assert_array_equal desired, result
    end
  end
end
