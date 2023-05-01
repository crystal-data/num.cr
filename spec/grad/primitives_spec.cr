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

describe Num::Grad::Context do
  it "creates an empty context" do
    ctx = Num::Grad::Context(Float32Tensor).new
    ctx.size.should eq 0
  end

  it "cannot pop a node from an empty context" do
    ctx = Num::Grad::Context(Float32Tensor).new
    expect_raises(IndexError) do
      ctx.pop
    end
  end

  it "can create a variable" do
    ctx = Num::Grad::Context(Float32Tensor).new
    t = Float32Tensor.full([3, 3], 3.5_f32)
    t_var = ctx.variable(t)
    t_var.context.should eq ctx
  end

  it "can create a variable with scalar" do
    ctx = Num::Grad::Context(Float32Tensor).new
    t = 3.14_f32
    t_var = ctx.variable(t)
    t_var.context.should eq ctx
    t_var.value[0].should eq t   # has the scalar
    t_var.value.size.should eq 1 # has only one element
  end
end

describe Num::Grad do
  it "can register a node" do
    ctx = Num::Grad::Context(Float32Tensor).new

    a = Float32Tensor.ones([3, 3])
    b = Float32Tensor.ones([3, 3])

    a_var = ctx.variable(a)
    b_var = ctx.variable(b)

    result = a_var + b_var

    ctx.size.should eq 1
  end
end
