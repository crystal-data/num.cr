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

# :nodoc:
class Num::SigmoidCrossEntropyBackwardKernel(T) < Num::Kernel(T)
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
                __global const #{dtype} * restrict const A_data,
                __global const int * restrict B_shape,
                __global const int * restrict B_strides,
                const int B_offset,
                __global const #{dtype} * restrict const B_data,
                __global const int * restrict C_shape,
                __global const int * restrict C_strides,
                const int C_offset,
                __global const #{dtype} * restrict const C_data,
                const int batch_size
                )
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        const int C_real_idx = opencl_getIndexOfElementID(rank, C_shape, C_strides, C_offset, elemID);
        dst_data[dst_real_idx] = A_data[A_real_idx] * (((#{dtype})1 / ((#{dtype})1 + exp(-B_data[B_real_idx]))) - C_data[C_real_idx]) / (#{dtype})batch_size;
      }
    }
    "
  end

  def call(
    gradient : Tensor(U, OCL(U)),
    cache : Tensor(U, OCL(U)),
    target : Tensor(U, OCL(U))
  ) : Tensor(U, OCL(U)) forall U
    batch_size = cache.shape[0]
    result = Tensor(U, OCL(U)).zeros_like(cache)
    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      cache.data.shape, cache.data.strides, cache.offset, cache.data.to_unsafe,
      target.data.shape, target.data.strides, target.offset, target.data.to_unsafe,
      batch_size
    )

    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

create_kernel_children(SigmoidCrossEntropyBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::SigmoidCrossEntropyKernel(T) < Num::Kernel(T)
  def call(a : Tensor(T, OCL(T)), b : Tensor(T, OCL(T)))
    result = Tensor(T, OCL(T)).zeros_like(a)

    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe,
      b.data.shape, b.data.strides, b.offset, b.data.to_unsafe,
    )

    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end

  def get_program(dtype)
    "
    #{super}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict C_shape,
        __global const int * restrict C_strides,
        const int C_offset,
        __global #{dtype} * restrict C,
        __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{dtype} * restrict const A,
        __global const int * restrict B_shape,
                __global const int * restrict B_strides,
                const int B_offset,
                __global const #{dtype} * restrict const B) {
              for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
                const int c = opencl_getIndexOfElementID(rank, C_shape, C_strides, C_offset, elemID);
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        C[c] = -B[b] * A[a] + max(A[a], (#{dtype})0) + log1p(exp(-fabs(A[a])));
      }
    }
    "
  end
end

create_kernel_children(SigmoidCrossEntropyKernel, [Float32, Float64])
