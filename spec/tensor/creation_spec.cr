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
  it "Converts to and from Array" do
    [1, 2, 3].to_tensor.to_a.should eq [1, 2, 3]
  end

  it "Converts enumerables to Tensor" do
    a = Set{1, 2, 3}.to_tensor
    assert_array_equal a, [1, 2, 3].to_tensor
  end

  it "Leaves Tensor as Tensor" do
    a = [1, 2, 3].to_tensor.to_tensor
    a.is_a?(Tensor(Int32)).should be_true
  end

  it "Creates a Tensor of zeros" do
    expected = [0, 0, 0].to_tensor
    result = Tensor(Int32).zeros([3])
    assert_array_equal expected, result
  end

  it "Creates a Tensor of ones" do
    expected = [1, 1, 1].to_tensor
    result = Tensor(Int32).ones([3])
    assert_array_equal expected, result
  end

  it "Creates a ranged Tensor" do
    expected = [0, 1, 2].to_tensor
    result = Tensor.range(3)
    assert_array_equal expected, result
  end

  it "Creates a ranged Tensor between values" do
    expected = [1, 2, 3].to_tensor
    result = Tensor.range(1, 4)
    assert_array_equal expected, result
  end

  it "Creates a ranged Tensor with step" do
    expected = [0, 2, 4].to_tensor
    result = Tensor.range(0, 5, 2)
    assert_array_equal expected, result
  end

  it "Creates an eye matrix" do
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].to_tensor
    result = Tensor(Int32).eye(3)
    assert_array_equal expected, result
  end

  it "Creates an eye matrix with offset" do
    expected = [[0, 0, 0], [1, 0, 0], [0, 1, 0]].to_tensor
    result = Tensor(Int32).eye(3, offset: -1)
    assert_array_equal expected, result
  end

  it "Creates an identity matrix" do
    expected = [[1, 0], [0, 1]].to_tensor
    result = Tensor(Int32).identity(2)
    assert_array_equal expected, result
  end

  it "Puts a Tensor along the diagonal" do
    expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]].to_tensor
    result = Tensor.diag([1, 2, 3])
    assert_array_equal expected, result
  end

  it "Creates a vandermonde matrix" do
    expected = [[1, 1, 1], [4, 2, 1], [9, 3, 1]].to_tensor
    result = Tensor.vandermonde([1, 2, 3], 3)
    assert_array_equal expected, result
  end

  it "Creates a Tensor from a shape" do
    expected = [0, 0, 0].to_tensor
    result = Tensor(Int32).new([3])
    assert_array_equal expected, result
  end

  it "Creates a Tensor from a block" do
    expected = [0, 1, 2].to_tensor
    result = Tensor.new([3]) { |i| i }
    assert_array_equal expected, result
  end
end
