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
  it "backpropogates for addition" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([1.0_f32])
    b = ctx.variable([1.0_f32])

    result = a + b
    result.backprop

    expected = [1_f32].to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
  end

  it "backpropogates for addition opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    a = ctx.variable([1.0_f32].to_tensor(OCL))
    b = ctx.variable([1.0_f32].to_tensor(OCL))

    result = a + b
    result.backprop

    expected = [1_f32].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
  end

  it "backpropogates for subtraction" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([1.0_f32])
    b = ctx.variable([1.0_f32])

    result = a - b
    result.backprop

    expected = [1_f32].to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
  end

  it "backpropogates for subtraction opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    a = ctx.variable([1.0_f32].to_tensor(OCL))
    b = ctx.variable([1.0_f32].to_tensor(OCL))

    result = a - b
    result.backprop

    expected = [1_f32].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
  end

  it "backpropogates for multiplication" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([3.0_f32])
    b = ctx.variable([2.0_f32])

    result = a * b
    result.backprop

    expected = [2_f32].to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
  end

  it "backpropogates for multiplication opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    a = ctx.variable([3.0_f32].to_tensor(OCL))
    b = ctx.variable([2.0_f32].to_tensor(OCL))

    result = a * b
    result.backprop

    expected = [2_f32].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
  end

  it "backpropogates for division" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([3.0_f32])
    b = ctx.variable([2.0_f32])

    result = a / b
    result.backprop

    expected = [0.5_f32].to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
  end

  it "backpropogates for division opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    a = ctx.variable([3.0_f32].to_tensor(OCL))
    b = ctx.variable([2.0_f32].to_tensor(OCL))

    result = a / b
    result.backprop

    expected = [0.5_f32].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
  end

  it "backpropogates for exponentiation" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = ctx.variable([3.0_f32])
    b = ctx.variable([2.0_f32])

    result = a ** b
    result.backprop

    expected = [6_f32].to_tensor

    Num::Testing.tensor_equal(a.grad, expected).should be_true
  end

  it "backpropogates for exponentiation opencl", tags: "opencl" do
    ctx = Num::Grad::Context(Float32ClTensor).new

    a = ctx.variable([3.0_f32].to_tensor(OCL))
    b = ctx.variable([2.0_f32].to_tensor(OCL))

    result = a ** b
    result.backprop

    expected = [6_f32].to_tensor

    Num::Testing.tensor_equal(a.grad.cpu, expected).should be_true
  end
end
