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
  it "reads a tensor from a file" do
    result = Tensor(Int32, CPU(Int32)).from_npy("#{__DIR__}/data/5x5.npy")
    expected = Tensor.new([5, 5]) { |i| i }
    Num::Testing.tensor_equal(result, expected).should be_true
  end

  it "reads a rank 1 tensor from a file" do
    result = Tensor(Int32, CPU(Int32)).from_npy("#{__DIR__}/data/single_rank_tensor.npy")
    expected = [1, 2, 3, 4].to_tensor
    Num::Testing.tensor_equal(result, expected).should be_true
  end
end
