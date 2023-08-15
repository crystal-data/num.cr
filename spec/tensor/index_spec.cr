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
require "../spec_helper"

describe Tensor do
  it "slices a Tensor using valid indexers" do
    a = Tensor.new([3, 3, 3]) { |i| i }

    r0 = a[0]
    e0 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]].to_tensor
    Num::Testing.tensor_equal(r0, e0).should be_true

    r1 = a[..., 1]
    e1 = [[3, 4, 5], [12, 13, 14], [21, 22, 23]].to_tensor
    Num::Testing.tensor_equal(r1, e1).should be_true

    r2 = a[{1..., 2}]
    e2 = [[[9, 10, 11], [12, 13, 14], [15, 16, 17]]].to_tensor
    Num::Testing.tensor_equal(r2, e2).should be_true
  end

  it "catches an out of bound indexer" do
    a = [[1, 2], [3, 4]].to_tensor
    expect_raises(Num::Exceptions::IndexError) do
      a[3]
    end

    expect_raises(Num::Exceptions::IndexError) do
      a[3...]
    end

    expect_raises(Num::Exceptions::IndexError) do
      a[{3..., 1}]
    end
  end

  it "indexes a 1D tensor using integer tensor" do
    x = [10, 20, 30, 40, 50].to_tensor
    indices = [1, 3].to_tensor
    y = x[indices]
    Num::Testing.tensor_equal(y, [20, 40].to_tensor)
  end

  it "indexes multidimensional tensor using integer tensor" do
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].to_tensor
    indices = [0, 2].to_tensor
    y = x[indices]
    Num::Testing.tensor_equal(y, [[1, 2, 3], [7, 8, 9]].to_tensor)
  end

  it "returns a tensor with an additional dimension when indexing with another tensor" do
    # Create tensor 'c' with shape [4, 2]
    c = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ].to_tensor

    # Create tensor 'x' with shape [6, 3], filled with valid indices for 'c'
    # indices are modulo 4 to stay within range
    x = [
      [0, 3, 2],
      [1, 2, 2],
      [0, 3, 2],
      [1, 0, 1],
      [2, 3, 2],
      [0, 0, 1],
    ].to_tensor

    # Perform indexing
    result = c[x]

    # The resulting tensor should have shape [6, 3, 2]
    result.shape.should eq([6, 3, 2])
    Num::Testing.tensor_equal(result, [
      [[0, 0], [1, 1], [1, 0]],
      [[0, 1], [1, 0], [1, 0]],
      [[0, 0], [1, 1], [1, 0]],
      [[0, 1], [0, 0], [0, 1]],
      [[1, 0], [1, 1], [1, 0]],
      [[0, 0], [0, 0], [0, 1]],
    ].to_tensor)
  end
end
