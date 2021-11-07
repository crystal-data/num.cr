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
  it "broadcasts a Tensor to a new shape" do
    a = [0, 1, 2].to_tensor

    expected = [[0, 1, 2], [0, 1, 2], [0, 1, 2]].to_tensor
    result = a.broadcast_to([3, 3])

    Num::Testing.tensor_equal(result, expected)
  end

  it "can't broadcast a Tensor to an invalid shape" do
    a = [0, 1, 2].to_tensor
    expect_raises(Num::Exceptions::ValueError) do
      a.broadcast_to([3, 5])
    end
  end

  it "broadcasts two Tensors against each other" do
    a = Tensor.new([3, 3]) { |i| i }
    b = [0, 1, 2].to_tensor

    a, b = a.broadcast(b)

    a.shape.should eq b.shape
  end

  it "cannot broadcast invalid shapes against each other" do
    a = Tensor.new([3, 3]) { |i| i }
    b = Tensor.new([4]) { |i| i }

    expect_raises(Num::Exceptions::ValueError) do
      a.broadcast(b)
    end
  end

  it "reshapes a Tensor" do
    a = Tensor.new([3, 3, 3]) { |i| i }
    b = Tensor.new([9, 3]) { |i| i }

    Num::Testing.tensor_equal(a.reshape(b.shape), b).should be_true
  end

  it "reshapes a Tensor using an unknown dimension" do
    a = Tensor.new([3, 3, 3]) { |i| i }
    b = Tensor.new([9, 3]) { |i| i }

    Num::Testing.tensor_equal(a.reshape(-1, 3), b).should be_true
  end

  it "cannot reshape into an invalid shape" do
    a = Tensor.new([3, 3, 3]) { |i| i }
    expect_raises(Num::Exceptions::ValueError) do
      a.reshape(10, 10)
    end
  end
end
