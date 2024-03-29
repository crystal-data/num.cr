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
  # :nodoc:
  def subtract_backward(gradient : Tensor(U, OCL(U))) : Array(Tensor(U, OCL(U))) forall U
    negative_one = [U.new(-1)].to_tensor(OCL)
    [gradient, gradient * negative_one]
  end

  # :nodoc:
  def divide_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U))),
    bv : Variable(Tensor(U, OCL(U)))
  ) : Array(Tensor(U, OCL(U))) forall U
    a = av.value
    b = bv.value

    r0 = gradient / b
    r1 = call_opencl_kernel(
      U,
      DivideBackwardsTwoKernel,
      [Float32, Float64],
      gradient, a, b
    )
    [r0, r1]
  end

  # :nodoc:
  def power_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U))),
    bv : Variable(Tensor(U, OCL(U)))
  ) forall U
    r0 = call_opencl_kernel(
      U,
      PowerBackwardsOneKernel,
      [Float32, Float64],
      gradient, av.value, bv.value
    )
    r1 = call_opencl_kernel(
      U,
      PowerBackwardsTwoKernel,
      [Float32, Float64],
      gradient, av.value, bv.value
    )

    [r0, r1]
  end

  # :nodoc:
  def exp_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      ExpBackwardsKernel,
      [Float32, Float64],
      gradient, av.value
    )
    [result]
  end

  # :nodoc:
  def log_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      LogBackwardsKernel,
      [Float32, Float64],
      gradient, av.value
    )
    [result]
  end

  # :nodoc:
  def sin_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      SinBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def cos_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      CosBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def tan_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      TanBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def tanh_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      TanhGradBackwardsKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def asin_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      AsinBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def acos_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      AcosBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def atan_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = call_opencl_kernel(
      U,
      AtanBackwardKernel,
      [Float32, Float64],
      gradient, a.value
    )
    [result]
  end

  # :nodoc:
  def sum_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    # Tensor#dup not available for OCL Tensors, so adding 0 to create copy for now
    [gradient + U.new(0)]
  end
end
