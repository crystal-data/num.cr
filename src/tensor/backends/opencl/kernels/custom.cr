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
class Num::TranposeKernel(T) < Num::Kernel(T)
  def get_program(dtype)
    "
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void #{@@name}(
    const int heightA, const int widthA,
    	__global const #{dtype} *a,
    	__global #{dtype} *a_T)
    {

    	const int colA = get_global_id(0);

    	for (int rowA = 0; rowA < heightA; rowA++)
    	{
    		a_T[colA *heightA + rowA] = a[rowA *widthA + colA];
    	}
    }
    "
  end

  def call(a : Tensor(T, OCL(T))) : Tensor(T, OCL(T))
    unless a.rank == 2
      raise Num::Exceptions::ValueError.new("Only CLTensors of rank 2 can be transposed")
    end
    m, n = a.shape
    result = Tensor(T, OCL(T)).new([n, m])
    Cl.args(@kernel, m, n, a.to_unsafe, result.to_unsafe)
    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

# :nodoc:
class Num::Int32TransposeKernel < Num::TranposeKernel(Int32)
  @@name = "transpose"
end

# :nodoc:
class Num::UInt32TransposeKernel < Num::TranposeKernel(UInt32)
  @@name = "transpose"
end

# :nodoc:
class Num::Float32TransposeKernel < Num::TranposeKernel(Float32)
  @@name = "transpose"
end

# :nodoc:
class Num::Float64TransposeKernel < Num::TranposeKernel(Float64)
  @@name = "transpose"
end

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

# :nodoc:
class Num::Float32SigmoidCrossEntropyBackwardKernel < Num::SigmoidCrossEntropyBackwardKernel(Float32)
  @@name = "sigmoidCrossEntropyBackward"
end

# :nodoc:
class Num::Float64SigmoidCrossEntropyBackwardKernel < Num::SigmoidCrossEntropyBackwardKernel(Float32)
  @@name = "sigmoidCrossEntropyBackward"
end

module Num
  def sgd_optimizer_kernel
    kernel = "
    #{index_of_element}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void sgdOptimize
                (const int rank,
                const int len,
                __global const int * restrict dst_shape,
                __global const int * restrict dst_strides,
                const int dst_offset,
                __global       float * dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const float * restrict const A_data,
                const float learning_rate
                )
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        dst_data[dst_real_idx] = dst_data[dst_real_idx] - learning_rate * A_data[A_real_idx];
      }
    }
    "
    program = Cl.create_and_build(
      Num::ClContext.instance.context,
      kernel, Num::ClContext.instance.device
    )
    {% if flag?(:debugcl) %}
      puts Cl.build_errors(program, [Num::ClContext.instance.device])
    {% end %}
    Cl.create_kernel(program, "sgdOptimize")
  end
end
