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

module Num::Grad
  def subtract_backward(gradient : Tensor(U, OCL(U))) : Array(Tensor(U, OCL(U))) forall U
    negative_one = [U.new(-1)].to_tensor(OCL)
    [gradient, gradient * negative_one]
  end

  private def two_variable_op_call(fn, gradient : U, a : U, b : U) : U forall U
    result = U.zeros_like(a)

    Cl.args(
      fn, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe,
      b.data.shape, b.data.strides, b.offset, b.data.to_unsafe,
    )
    Cl.run(Num::ClContext.instance.queue, fn, result.size)
    result
  end

  def divide_backward(
    gradient : Tensor(Float32, OCL(Float32)),
    av : Variable(Tensor(Float32, OCL(Float32))),
    bv : Variable(Tensor(Float32, OCL(Float32)))
  ) : Array(Tensor(Float32, OCL(Float32))) forall U
    a = av.value
    b = bv.value

    r0 = gradient / b
    r1 = two_variable_op_call(
      Num::OpenCLKernelCache.divideBackwardsTwoFloat,
      gradient,
      a,
      b
    )

    [r0, r1]
  end

  def power_backward(
    gradient : Tensor(Float32, OCL(Float32)),
    av : Variable(Tensor(Float32, OCL(Float32))),
    bv : Variable(Tensor(Float32, OCL(Float32)))
  )
    r0 = two_variable_op_call(
      Num::OpenCLKernelCache.powerBackwardsOneFloat,
      gradient,
      av.value,
      bv.value
    )
    r1 = two_variable_op_call(
      Num::OpenCLKernelCache.powerBackwardsTwoFloat,
      gradient,
      av.value,
      bv.value
    )

    [r0, r1]
  end

  def exp_backward(
    gradient : Tensor(Float32, OCL(Float32)),
    av : Variable(Tensor(Float32, OCL(Float32)))
  )
    a = av.value
    prok = Num::OpenCLKernelCache.expBackwards
    result = gradient.class.new(gradient.shape)
    Cl.args(
      prok, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )
    Cl.run(Num::ClContext.instance.queue, fn, result.size)
    result
    [r0, r1]
  end
end
