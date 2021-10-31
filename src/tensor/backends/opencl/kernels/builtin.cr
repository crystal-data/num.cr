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

class Num::BuiltinKernel(T) < Num::Kernel(T)
  @@fn : String = ""
  @@name : String = ""

  def initialize
    {% if T != Float32 && T != Float64 %}
      {% raise "Builtin OpenCL functions only accept doubles and floats" %}
    {% end %}
    super
  end

  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void #{@@name}
                (const int rank,
                const int len,
                __global const int * restrict dst_shape,
                __global const int * restrict dst_strides,
                const int dst_offset,
                __global       #{dtype} * restrict const dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{dtype} * restrict const A_data)
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        dst_data[dst_real_idx] = #{@@fn}(A_data[A_real_idx]);
      }
    }
    "
  end

  def call(a : Tensor(T, OCL(T)))
    result = Tensor(T, OCL(T)).new(a.shape)
    Cl.args(
      @kernel,
      result.rank,
      result.size,
      result.data.shape,
      result.data.strides,
      result.offset,
      result.to_unsafe,
      a.data.shape,
      a.data.strides,
      a.offset,
      a.to_unsafe
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

macro generate_builtin_kernels(cls, fn)
  class Num::{{ cls }}(T) < Num::BuiltinKernel(T)
    @@fn = "{{ fn.id }}"
    @@name = "{{ fn.id }}Kernel"
  end
end

generate_builtin_kernels(AcosKernel, :acos)
generate_builtin_kernels(AcoshKernel, :acosh)
generate_builtin_kernels(AcospiKernel, :acospi)
generate_builtin_kernels(AsinKernel, :asin)
generate_builtin_kernels(AsinhKernel, :asinh)
generate_builtin_kernels(AsinpiKernel, :asinpi)
generate_builtin_kernels(AtanKernel, :atan)
generate_builtin_kernels(AtanhKernel, :atanh)
generate_builtin_kernels(AtanpiKernel, :atanpi)
generate_builtin_kernels(CbrtKernel, :cbrt)
generate_builtin_kernels(CeilKernel, :ceil)
generate_builtin_kernels(CosKernel, :cos)
generate_builtin_kernels(CoshKernel, :cosh)
generate_builtin_kernels(CospiKernel, :cospi)
generate_builtin_kernels(ErfcKernel, :erfc)
generate_builtin_kernels(ErfKernel, :erf)
generate_builtin_kernels(ExpKernel, :exp)
generate_builtin_kernels(Exp2Kernel, :exp2)
generate_builtin_kernels(Exp10Kernel, :exp10)
generate_builtin_kernels(Expm1Kernel, :expm1)
generate_builtin_kernels(FabsKernel, :fabs)
generate_builtin_kernels(FloorKernel, :floor)
generate_builtin_kernels(LgammaKernel, :lgamma)
generate_builtin_kernels(LogKernel, :log)
generate_builtin_kernels(Log2Kernel, :log2)
generate_builtin_kernels(Log10Kernel, :log10)
generate_builtin_kernels(Log1pKernel, :log1p)
generate_builtin_kernels(LogbKernel, :logb)
generate_builtin_kernels(RintKernel, :rint)
generate_builtin_kernels(RoundKernel, :round)
generate_builtin_kernels(RsqrtKernel, :rsqrt)
generate_builtin_kernels(SinKernel, :sin)
generate_builtin_kernels(SinhKernel, :sinh)
generate_builtin_kernels(SinpiKernel, :sinpi)
generate_builtin_kernels(SqrtKernel, :sqrt)
generate_builtin_kernels(TanKernel, :tan)
generate_builtin_kernels(TanhKernel, :tanh)
generate_builtin_kernels(TanpiKernel, :tanpi)
generate_builtin_kernels(TgammaKernel, :tgamma)
generate_builtin_kernels(TruncKernel, :trunc)
