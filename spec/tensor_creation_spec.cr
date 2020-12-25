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

require "./spec_helper"

describe Tensor do
  it "creates a tensor of zeros" do
    expected = [0, 0, 0]
    result = Tensor(Int32).zeros([3])
    expected.should eq result.to_a
  end

  it "creates a tensor of zeros like" do
    expected = [0, 0, 0, 0]
    t = Tensor(Int32).zeros([2, 2])
    v = Tensor(Int32).zeros_like(t)
    v.shape.should eq t.shape
    expected.should eq v.to_a
  end

  it "creates a tensor of ones" do
    expected = [1, 1, 1]
    result = Tensor(Int32).ones([3])
    expected.should eq result.to_a
  end

  it "creates a tensor of ones like" do
    expected = [1, 1, 1, 1]
    t = Tensor(Int32).ones([2, 2])
    v = Tensor(Int32).ones_like(t)
    v.shape.should eq t.shape
    expected.should eq v.to_a
  end

  it "creates a tensor of value" do
    expected = [2, 2, 2]
    result = Tensor.full([3], 2)
    expected.should eq result.to_a
  end

  it "creates a tensor of ones like" do
    expected = [2, 2, 2, 2]
    t = Tensor(Int32).ones([2, 2])
    v = Tensor.full_like(t, 2)
    v.shape.should eq t.shape
    expected.should eq v.to_a
  end
end
