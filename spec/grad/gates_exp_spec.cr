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
  it "backpropogates for exp" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = Tensor.new([10], device: CPU) { |i| i.to_f32 / 10 }
    x = ctx.variable(a)

    result = x.exp
    result.backprop

    expected = [1, 1.10517, 1.2214, 1.34986, 1.49182, 1.64872, 1.82212, 2.01375, 2.22554, 2.4596].to_tensor
    Num::Testing.tensor_equal(x.grad, expected, tolerance: 1e-3).should be_true
  end

  it "backpropogates for exp opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

    a = Tensor.new([10], device: OCL) { |i| i.to_f32 / 10 }
    x = ctx.variable(a)

    result = x.exp
    result.backprop

    expected = [1, 1.10517, 1.2214, 1.34986, 1.49182, 1.64872, 1.82212, 2.01375, 2.22554, 2.4596].to_tensor
    Num::Testing.tensor_equal(x.grad.cpu, expected, tolerance: 1e-3).should be_true
  end
end
