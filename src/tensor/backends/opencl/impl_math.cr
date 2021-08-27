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

module Num
  macro elementwise_op(fn, dtype, prefix)
    def {{fn.id}}(a : Tensor({{dtype}}, OCL({{dtype}})), b : Tensor({{dtype}}, OCL({{dtype}})))
      prok = OpenCLKernelCache.{{prefix}}{{fn.id}}_ew
      a, b = a.broadcast(b)
      t = Tensor({{dtype}}, OCL({{dtype}})).new(a.shape)
      Cl.args(prok, t.rank, t.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe, b.data.shape, b.data.strides, b.offset, b.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor({{dtype}}, OCL({{dtype}})), b : Tensor({{dtype}}, OCL({{dtype}})))
      prok = OpenCLKernelCache.s{{fn.id}}_ew_inpl
      b = b.broadcast_to(a.shape)
      Cl.args(prok, a.rank, a.size, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe, b.data.shape, b.data.strides, b.offset, b.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, a.size)
    end
  end

  elementwise_op add, Float32, s
  elementwise_op add, Float64, d
  elementwise_op subtract, Float32, s
  elementwise_op subtract, Float64, d
  elementwise_op multiply, Float32, s
  elementwise_op multiply, Float64, d
  elementwise_op divide, Float32, s
  elementwise_op divide, Float64, d
  elementwise_op power, Float32, s
  elementwise_op power, Float64, d

  macro builtin(fn)
    def {{fn.id}}(a : Tensor(Float32, OCL(Float32)))
      prok = OpenCLKernelCache.s{{fn.id}}_ew_fn
      t = Tensor(Float32, OCL(Float32)).new(a.shape)
      Cl.args(prok, a.rank, a.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor(Float32, OCL(Float32)))
      prok = OpenCLKernelCache.s{{fn.id}}_ew_fn_inpl
      Cl.args(prok, a.rank, a.size, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok)
    end

    def {{fn.id}}(a : Tensor(Float64, OCL(Float64)))
      prok = OpenCLKernelCache.d{{fn.id}}_ew_fn
      t = Tensor(Float32, OCL(Float32)).new(a.shape)
      Cl.args(prok, a.rank, a.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor(Float64, OCL(Float64)))
      prok = OpenCLKernelCache.d{{fn.id}}_ew_fn_inpl
      Cl.args(prok, a.rank, a.size, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok)
    end
  end

  builtin acospi
  builtin asin
  builtin asinh
  builtin asinpi
  builtin atan
  builtin atanh
  builtin atanpi
  builtin cbrt
  builtin ceil
  builtin cos
  builtin cosh
  builtin cospi
  builtin erfc
  builtin erf
  builtin exp
  builtin exp2
  builtin exp10
  builtin expm1
  builtin fabs
  builtin floor
  builtin lgamma
  builtin log
  builtin log2
  builtin log10
  builtin log1p
  builtin logb
  builtin rint
  builtin round
  builtin sqrt
  builtin rsqrt
  builtin sin
  builtin sinh
  builtin sinpi
  builtin tan
  builtin tanh
  builtin tanpi
  builtin tgamma
  builtin trunc
end
