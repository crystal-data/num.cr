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

describe Num::Grad do
  it "backpropogates for matrix multiplication" do
    ctx = Num::Grad::Context(Float32Tensor).new

    at = [[1, 2] of Float32, [3, 4] of Float32].to_tensor
    bt = [[1, 2] of Float32, [3, 4] of Float32].to_tensor

    a = ctx.variable(at)
    b = ctx.variable(bt)

    result = a.matmul(b)
    result.backprop

    expected = [[3, 7], [3, 7]].to_tensor

    Num::Testing.tensor_equal(a.grad, expected)
  end

  it "backpropogates for matrix multiplication opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    at = [[1, 2] of Float32, [3, 4] of Float32].to_tensor(OCL)
    bt = [[1, 2] of Float32, [3, 4] of Float32].to_tensor(OCL)

    a = ctx.variable(at)
    b = ctx.variable(bt)

    result = a.matmul(b)
    result.backprop

    expected = [[3, 7], [3, 7]].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected)
  end
end
