# Copyright (c) 2021 Crystal Data Contributors
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

describe Tensor do
  it "converts a flat Tensor to an array" do
    a = [1, 2, 3].to_tensor
    b = a.to_a
    b.should eq [1, 2, 3]
  end

  it "converts a multi-dimensional Tensor to an array" do
    a = Tensor.new([2, 2, 2]) { |i| i }
    b = a.to_a
    b.should eq [0, 1, 2, 3, 4, 5, 6, 7]
  end

  it "converts a sliced Tensor to an array" do
    a = Tensor.new([10]) { |i| i }
    b = a[{..., 2}]
    result = b.to_a
    result.should eq [0, 2, 4, 6, 8]
  end

  it "changes the type of a Tensor" do
    a = [3.4, 4.15, 7.6].to_tensor
    result = a.as_type(Int32)
    expected = [3, 4, 7].to_tensor
    Num::Testing.tensor_equal(result, expected).should be_true
  end
end
