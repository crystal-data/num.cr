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
class Num::SGDOptimizeKernel(T) < Num::Kernel(T)
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
                __global       #{dtype} * dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{dtype} * restrict const A_data,
                const #{dtype} learning_rate
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
  end

  def call(value : Tensor(U, OCL(U)), gradient : Tensor(U, OCL(U)), learning_rate : Float) forall U
    Cl.args(
      @kernel, value.rank, value.size,
      value.data.shape, value.data.strides, value.offset, value.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      U.new(learning_rate),
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, value.size)
    nil
  end
end

# :nodoc:
class Num::Float32SGDOptimizeKernel(T) < Num::SGDOptimizeKernel(Float32)
  @@name = "sgdOptimize"
end

# :nodoc:
class Num::Float64SGDOptimizeKernel(T) < Num::SGDOptimizeKernel(Float64)
  @@name = "sgdOptimize"
end
