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
  it "backpropogates for log" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = Tensor.new([10], device: CPU) { |i| (i + 1).to_f32 / 10 }
    x = ctx.variable(a)

    result = x.log
    result.backprop

    expected = [10.0000, 5.0000, 3.3333, 2.5000, 2.0000, 1.6667, 1.4286, 1.2500, 1.1111, 1.0000].to_tensor
    Num::Testing.tensor_equal(x.grad, expected, tolerance: 1e-3).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for log opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = Tensor.new([10], device: OCL) { |i| (i + 1).to_f32 / 10 }
      x = ctx.variable(a)

      result = x.log
      result.backprop

      expected = [10.0000, 5.0000, 3.3333, 2.5000, 2.0000, 1.6667, 1.4286, 1.2500, 1.1111, 1.0000].to_tensor
      Num::Testing.tensor_equal(x.grad.cpu, expected, tolerance: 1e-3).should be_true
    end
  {% end %}
end
