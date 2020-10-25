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

describe Tensor do
  it "sorts a one dimensional Tensor" do
    a = [4, 3, 2, 1].to_tensor
    result = Num.sort(a)
    expected = [1, 2, 3, 4].to_tensor
    assert_array_equal(result, expected)
  end

  it "sorts a strided Tensor" do
    a = [4, 3, 2, 1].to_tensor[{..., 2}]
    result = Num.sort(a)
    expected = [2, 4]
    assert_array_equal(result, expected)
  end

  it "sorts a Tensor along an axis" do
    a = [[3, 5, 6], [1, 1, 2], [9, 2, 3]].to_tensor
    result = Num.sort(a, 0)
    expected = [[1, 1, 2], [3, 2, 3], [9, 5, 6]].to_tensor
    assert_array_equal(result, expected)
  end

  it "sorts a strided Tensor along an axis" do
    a = [[3, 4, 5, 1], [2, 1, 3, 2], [4, 7, 6, 2]].to_tensor[..., {..., 2}]
    result = Num.sort(a, 0)
    expected = [[2, 3], [3, 5], [4, 6]].to_tensor
    assert_array_equal(result, expected)
  end

  it "finds the argmax of a 1d tensor" do
    a = [1, 2, 3, 2, 1].to_tensor
    a.argmax.should eq 2
  end

  it "finds the argmax of an nd tensor" do
    a = [
      [[3, 9], [8, 3], [4, 8]],
      [[3, 7], [8, 7], [4, 6]],
      [[1, 8], [8, 1], [6, 7]],
    ]
    result = Num.argmax(a, 1)
    expected = [[1, 0], [1, 0], [1, 0]].to_tensor
    assert_array_equal(result, expected)
  end

  it "finds the argmin of a 1d tensor" do
    a = [1, 2, 3, 2, 1].to_tensor
    a.argmin.should eq 0
  end

  it "finds the argmin of an nd tensor" do
    a = [
      [[3, 9], [8, 3], [4, 8]],
      [[3, 7], [8, 7], [4, 6]],
      [[1, 8], [8, 1], [6, 7]],
    ]
    result = Num.argmin(a, 1)
    expected = [[0, 1], [0, 2], [0, 1]].to_tensor
    assert_array_equal(result, expected)
  end
end
