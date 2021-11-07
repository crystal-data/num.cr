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
  it "backpropogates for sin" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [1_f32].to_tensor
    x = ctx.variable(a)

    result = x.sin
    result.backprop

    expected = [Math.cos(1)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for sin opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [1_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.sin
      result.backprop

      expected = [Math.cos(1)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for cos" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [1_f32].to_tensor
    x = ctx.variable(a)

    result = x.cos
    result.backprop

    expected = [-Math.sin(1)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for cos opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [1_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.cos
      result.backprop

      expected = [-Math.sin(1)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for tan" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [1_f32].to_tensor
    x = ctx.variable(a)

    result = x.tan
    result.backprop

    expected = [1 / (Math.cos(1) ** 2)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for tan opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [1_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.tan
      result.backprop

      expected = [1 / (Math.cos(1) ** 2)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for asin" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [0.5.to_f32].to_tensor
    x = ctx.variable(a)

    result = x.asin
    result.backprop

    expected = [1 / Math.sqrt(1 - 0.5 ** 2)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for asin opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [0.5_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.asin
      result.backprop

      expected = [1 / Math.sqrt(1 - 0.5 ** 2)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for acos" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [0.5.to_f32].to_tensor
    x = ctx.variable(a)

    result = x.acos
    result.backprop

    expected = [-1 / Math.sqrt(1 - 0.5 ** 2)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for acos opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [0.5_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.acos
      result.backprop

      expected = [-1 / Math.sqrt(1 - 0.5 ** 2)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}

  it "backpropogates for atan" do
    ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

    a = [0.5.to_f32].to_tensor
    x = ctx.variable(a)

    result = x.atan
    result.backprop

    expected = [1 / (1 + 0.5 ** 2)].to_tensor

    Num::Testing.tensor_equal(x.grad, expected).should be_true
  end

  {% if flag?(:opencl) %}
    it "backpropogates for atan opencl", tags: "opencl" do
      ctx = Num::Grad::Context(Tensor(Float32, OCL(Float32))).new

      a = [0.5_f32].to_tensor(OCL)
      x = ctx.variable(a)

      result = x.atan
      result.backprop

      expected = [1 / (1 + 0.5 ** 2)].to_tensor

      Num::Testing.tensor_equal(x.grad.cpu, expected).should be_true
    end
  {% end %}
end
