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
  it "creates empty Tensor from shape" do
    result = Tensor(Int8, CPU(Int8)).new([2, 2])
    result.shape.should eq [2, 2]
    result.strides.should eq [2, 1]
  end

  it "creates fortran Tensor from shape and order" do
    result = Tensor(Int8, CPU(Int8)).new([2, 2], Num::ColMajor)
    result.shape.should eq [2, 2]
    result.strides.should eq [1, 2]
  end

  it "creates a Tensor with an initial value" do
    result = Tensor.new([2, 2], 3.5)
    result.class.should eq Tensor(Float64, CPU(Float64))
    result.to_a.should eq [3.5, 3.5, 3.5, 3.5]
  end

  it "creates a Tensor from a block" do
    result = Tensor.new([2, 2]) { |i| i }
    result.class.should eq Tensor(Int32, CPU(Int32))
    result.to_a.should eq [0, 1, 2, 3]
  end

  it "creates a Matrix from a block" do
    result = Tensor.new(2, 2) { |i, j| i + j }
    result.class.should eq Tensor(Int32, CPU(Int32))
    result.to_a.should eq [0, 1, 1, 2]
  end

  it "creates a Tensor from a flat array" do
    result = [1, 2, 3].to_tensor
    result.shape.should eq [3]
  end

  it "creates a Tensor from a multidimensional array" do
    result = [[1, 2], [3, 4]].to_tensor
    result.shape.should eq [2, 2]
  end

  it "raises if an array is jagged" do
    expect_raises(Num::Exceptions::ValueError) do
      [[1, 2, 3], [4, 5]].to_tensor
    end
  end

  it "creates Tensors from zeros and ones" do
    a = Tensor(Int32, CPU(Int32)).zeros([2, 2])
    b = Tensor(Int32, CPU(Int32)).zeros_like(a)
    c = Tensor(Int32, CPU(Int32)).ones([3, 3])
    d = Tensor(Int32, CPU(Int32)).ones_like(c)

    Num::Testing.tensor_equal(a, b).should be_true
    Num::Testing.tensor_equal(c, d).should be_true
  end

  it "creates a range Tensor" do
    result = Tensor.range(0, 5)
    result.shape.should eq [5]

    expected = [0, 1, 2, 3, 4].to_tensor
    Num::Testing.tensor_equal(result, expected).should be_true
  end

  it "creates a range Tensor with a step" do
    result = Tensor.range(0, 11, 2)
    result.shape.should eq [6]

    expected = [0, 2, 4, 6, 8, 10].to_tensor
    Num::Testing.tensor_equal(result, expected).should be_true
  end

  it "copies a Tensor" do
    a = [[1, 2], [3, 4]].to_tensor
    b = a.dup
    b[...] = 99
    Num::Testing.tensor_equal(a, b).should be_false
  end

  it "shallow copies a Tensor" do
    a = [[1, 2], [3, 4]].to_tensor
    b = a.view
    b[...] = 99
    Num::Testing.tensor_equal(a, b).should be_true
  end

  it "views a Tensor as a different type" do
    a = [0, 1, 2].to_tensor
    result = a.view(Int16)
    expected = [0, 0, 1, 0, 2, 0].to_tensor
    Num::Testing.tensor_equal(result, expected).should be_true
  end
end
