# Copyright (c) 2023 Crystal Data Contributors
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
  it "can sample a multinomial distribution" do
    input = [[0.4, 0.6], [0.5, 0.5]].to_tensor
    a = Tensor.multinomial(input, 5)
    b = [[0, 1, 0, 1, 1], [0, 1, 1, 0, 0]].to_tensor
    Num::Testing.tensor_equal(a, b).should be_true
  end

  it "can sample a multinomial distribution using a 1D input" do
    input = [0.2, 0.1, 0.3, 0.4].to_tensor
    a = Tensor.multinomial(input, 6)
    b = [2, 3, 3, 0, 3, 2].to_tensor
    Num::Testing.tensor_equal(a, b).should be_true
  end

  it "can sample a multinomial distribution using a non-normalized input" do
    input = [6, 3, 9, 12].to_tensor
    a = Tensor.multinomial(input, 6)
    b = [1, 2, 3, 3, 2, 2].to_tensor
    Num::Testing.tensor_equal(a, b).should be_true
  end
end
