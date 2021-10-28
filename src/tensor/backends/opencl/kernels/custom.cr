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
  # :nodoc:
  def transpose_kernel
    kernel = "
    __kernel void matrixTranspose(const int heightA, const int widthA,
    			__global
    	const float *a,
    		__global float *a_T)
    {

    	const int colA = get_global_id(0);

    	for (int rowA = 0; rowA < heightA; rowA++)
    	{
    		a_T[colA *heightA + rowA] = a[rowA *widthA + colA];
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
    Cl.create_kernel(program, "matrixTranspose")
  end

  def sigmoid_cross_entropy_backwards_kernel
    kernel = "
    #{index_of_element}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void sigmoidCrossEntropyBackward
                (const int rank,
                const int len,
                __global const int * restrict dst_shape,
                __global const int * restrict dst_strides,
                const int dst_offset,
                __global       float * restrict const dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const float * restrict const A_data,
                __global const int * restrict B_shape,
                __global const int * restrict B_strides,
                const int B_offset,
                __global const float * restrict const B_data,
                __global const int * restrict C_shape,
                __global const int * restrict C_strides,
                const int C_offset,
                __global const float * restrict const C_data,
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
        dst_data[dst_real_idx] = A_data[A_real_idx] * (((float)1 / ((float)1 + exp(-B_data[B_real_idx]))) - C_data[C_real_idx]) / (float)batch_size;
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
    Cl.create_kernel(program, "sigmoidCrossEntropyBackward")
  end

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
