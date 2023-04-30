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
  it "backpropogates for sum" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([1.0_f32, 0.9_f32, 0.8_f32, 0.7_f32, 0.6_f32,
                      0.5_f32, 0.4_f32, 0.3_f32, 0.2_f32, 0.1_f32])

    result = a.sum
    result.backprop

    expected = Array.new(10, 1.0_f32).to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
    # a.grad.to_a.should eq expected
  end

  {% if flag?(:opencl) %}
    it "backpropogates for sum opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Float32ClTensor).new

      a = ctx.variable([1.0_f32, 0.9_f32, 0.8_f32, 0.7_f32, 0.6_f32,
                        0.5_f32, 0.4_f32, 0.3_f32, 0.2_f32, 0.1_f32].to_tensor(OCL))

      result = a.sum
      result.backprop

      expected = Array.new(10, 1.0_f32).to_tensor

      Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for mean" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([1.0_f32, 0.9_f32, 0.8_f32, 0.7_f32, 0.6_f32,
                      0.5_f32, 0.4_f32, 0.3_f32, 0.2_f32, 0.1_f32])

    result = a.mean
    result.backprop

    expected = Array.new(10, 0.1_f32).to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
    # a.grad.to_a.should eq expected
  end

  {% if flag?(:opencl) %}
    it "backpropogates for mean opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Float32ClTensor).new

      a = ctx.variable([1.0_f32, 0.9_f32, 0.8_f32, 0.7_f32, 0.6_f32,
                        0.5_f32, 0.4_f32, 0.3_f32, 0.2_f32, 0.1_f32].to_tensor(OCL))

      result = a.mean
      result.backprop

      expected = Array.new(10, 0.1_f32).to_tensor

      Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
    end
  {% end %}
end
