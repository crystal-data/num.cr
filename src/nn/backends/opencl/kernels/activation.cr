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
class Num::ActivationBackwardKernel(T) < Num::Kernel(T)
  def call(gradient : Tensor(T, OCL(T)), a : Tensor(T, OCL(T)))
    result = Tensor(T, OCL(T)).zeros_like(a)
    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe,
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

# :nodoc:
class Num::ActivationForwardKernel(T) < Num::Kernel(T)
  def call(a : Tensor(T, OCL(T)))
    result = Tensor(T, OCL(T)).zeros_like(a)

    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )

    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

# :nodoc:
class Num::ActivationForwardInplaceKernel(T) < Num::Kernel(T)
  def call(a : Tensor(T, OCL(T)))
    Cl.args(
      @kernel, a.rank, a.size,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )

    Cl.run(Num::ClContext.instance.queue, @kernel, a.size)
    nil
  end
end

# :nodoc:
class Num::TanhBackwardKernel(T) < Num::ActivationBackwardKernel(T)
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

              C[c] = B[b] * ((#{dtype})1 - A[a] * A[a]);
            }
          }
    "
  end
end

create_kernel_children(TanhBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::SigmoidKernel(T) < Num::ActivationForwardKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict B_shape,
        __global const int * restrict B_strides,
        const int B_offset,
        __global #{dtype} * restrict B,
        __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{dtype} * restrict const A) {
              for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
                const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

        B[b] = (#{dtype})1 / ((#{dtype})1 + exp(-A[a]));
      }
    }
    "
  end
end

create_kernel_children(SigmoidKernel, [Float32, Float64])

# :nodoc:
class Num::SigmoidInplaceKernel(T) < Num::ActivationForwardInplaceKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict A_shape,
        __global const int * restrict A_strides,
        const int A_offset,
        __global #{dtype} * restrict A) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

        A[a] = (#{dtype})1 / ((#{dtype})1 + exp(-A[a]));
      }
    }
    "
  end
end

create_kernel_children(SigmoidInplaceKernel, [Float32, Float64])

# :nodoc:
class Num::SigmoidBackwardKernel(T) < Num::ActivationBackwardKernel(T)
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

        C[c] = A[a] * ((#{dtype})1 - A[a]) * B[b];
      }
    }
    "
  end
end

create_kernel_children(SigmoidBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::ReluKernel(T) < Num::ActivationForwardKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

     __kernel void #{@@name}(const int rank,
         const int len,
         __global const int * restrict B_shape,
         __global const int * restrict B_strides,
         const int B_offset,
         __global #{dtype} * restrict B,
         __global const int * restrict A_shape,
                 __global const int * restrict A_strides,
                 const int A_offset,
                 __global const #{dtype} * restrict const A) {
               for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
                 const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
         const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

         B[b] = max(A[a], (#{dtype})0);
       }
     }
    "
  end
end

create_kernel_children(ReluKernel, [Float32, Float64])

# :nodoc:
class Num::ReluInplaceKernel(T) < Num::ActivationForwardInplaceKernel(T)
  def get_program(dtype)
    "
    #{super}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict A_shape,
        __global const int * restrict A_strides,
        const int A_offset,
        __global #{dtype} * restrict A) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

        A[a] = max(A[a], (#{dtype})0);
      }
    }
    "
  end
end

create_kernel_children(ReluInplaceKernel, [Float32, Float64])

# :nodoc:
class Num::ReluBackwardKernel(T) < Num::ActivationBackwardKernel(T)
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

        C[c] = A[a] <= 0 ? (#{dtype})0 : B[b];
      }
    }
    "
  end
end

create_kernel_children(ReluBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::LeakyReluKernel(T) < Num::ActivationForwardKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

     __kernel void #{@@name}(const int rank,
         const int len,
         __global const int * restrict B_shape,
         __global const int * restrict B_strides,
         const int B_offset,
         __global #{dtype} * restrict B,
         __global const int * restrict A_shape,
                 __global const int * restrict A_strides,
                 const int A_offset,
                 __global const #{dtype} * restrict const A) {
               for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
                 const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
         const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

         B[b] = A[a] > 0 ? A[a] : A[a] * (#{dtype})0.01;
       }
     }
    "
  end
end

create_kernel_children(LeakyReluKernel, [Float32, Float64])

# :nodoc:
class Num::LeakyReluInplaceKernel(T) < Num::ActivationForwardInplaceKernel(T)
  def get_program(dtype)
    "
    #{super}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict A_shape,
        __global const int * restrict A_strides,
        const int A_offset,
        __global #{dtype} * restrict A) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);

        A[a] = A[a] > 0 ? A[a] : A[a] * (#{dtype})0.01;
      }
    }
    "
  end
end

create_kernel_children(LeakyReluInplaceKernel, [Float32, Float64])

# :nodoc:
class Num::LeakyReluBackwardKernel(T) < Num::ActivationBackwardKernel(T)
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

        C[c] = A[a] < 0 ? B[b] * (#{dtype})0.01 : B[b];
      }
    }
    "
  end
end

create_kernel_children(LeakyReluBackwardKernel, [Float32, Float64])
