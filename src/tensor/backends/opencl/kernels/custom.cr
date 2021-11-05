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
class Num::TransposeKernel(T) < Num::Kernel(T)
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

create_kernel_children(TransposeKernel, [Int32, UInt32, Float32, Float64])

# :nodoc:
class Num::BackwardKernel(T) < Num::Kernel(T)
  def call(gradient : Tensor(T, OCL(T)), a : Tensor(T, OCL(T)), b : Tensor(T, OCL(T)))
    result = Tensor(T, OCL(T)).zeros_like(a)

    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe,
      b.data.shape, b.data.strides, b.offset, b.data.to_unsafe,
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

# :nodoc:
class Num::DivideBackwardsTwoKernel(T) < Num::BackwardKernel(T)
  def get_program(dtype)
    "
    #{super}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
          __global
      const int * restrict D_shape,
        __global
      const int * restrict D_strides,
        const int D_offset,
          __global #{dtype} * restrict D,
          __global
      const int * restrict A_shape,
        __global
      const int * restrict A_strides,
        const int A_offset,
          __global
      const #{dtype} * restrict
      const A,
        __global
      const int * restrict B_shape,
        __global
      const int * restrict B_strides,
        const int B_offset,
          __global
      const #{dtype} * restrict
      const B,
        __global
      const int * restrict C_shape,
        __global
      const int * restrict C_strides,
        const int C_offset,
          __global
      const #{dtype} * restrict
      const C) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int d = opencl_getIndexOfElementID(rank, D_shape, D_strides, D_offset, elemID);
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        const int c = opencl_getIndexOfElementID(rank, C_shape, C_strides, C_offset, elemID);

        D[d] = -A[a] * B[b] / pow(C[c], 2);
      }
    }
    "
  end
end

create_kernel_children(DivideBackwardsTwoKernel, [Float32, Float64])

# :nodoc:
class Num::PowerBackwardsOneKernel(T) < Num::BackwardKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
          __global
      const int * restrict D_shape,
        __global
      const int * restrict D_strides,
        const int D_offset,
          __global #{dtype} * restrict D,
          __global
      const int * restrict A_shape,
        __global
      const int * restrict A_strides,
        const int A_offset,
          __global
      const #{dtype} * restrict
      const A,
        __global
      const int * restrict B_shape,
        __global
      const int * restrict B_strides,
        const int B_offset,
          __global
      const #{dtype} * restrict
      const B,
        __global
      const int * restrict C_shape,
        __global
      const int * restrict C_strides,
        const int C_offset,
          __global
      const #{dtype} * restrict
      const C) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int d = opencl_getIndexOfElementID(rank, D_shape, D_strides, D_offset, elemID);
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        const int c = opencl_getIndexOfElementID(rank, C_shape, C_strides, C_offset, elemID);

        D[d] = A[a] * C[c] * pow(B[b], C[c] == 0 ? (#{dtype}) 1 : C[c] - 1);
      }
    }
    "
  end
end

create_kernel_children(PowerBackwardsOneKernel, [Float32, Float64])

# :nodoc:
class Num::PowerBackwardsTwoKernel(T) < Num::BackwardKernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
          __global
      const int * restrict D_shape,
        __global
      const int * restrict D_strides,
        const int D_offset,
          __global #{dtype} * restrict D,
          __global
      const int * restrict A_shape,
        __global
      const int * restrict A_strides,
        const int A_offset,
          __global
      const #{dtype} * restrict
      const A,
        __global
      const int * restrict B_shape,
        __global
      const int * restrict B_strides,
        const int B_offset,
          __global
      const #{dtype} * restrict
      const B,
        __global
      const int * restrict C_shape,
        __global
      const int * restrict C_strides,
        const int C_offset,
          __global
      const #{dtype} * restrict
      const C) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int d = opencl_getIndexOfElementID(rank, D_shape, D_strides, D_offset, elemID);
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        const int c = opencl_getIndexOfElementID(rank, C_shape, C_strides, C_offset, elemID);

        D[d] = A[a] * pow(B[b], C[c]) * log(B[b] == 0 ? (#{dtype}) 1 : B[b]);
      }
    }
    "
  end
end

create_kernel_children(PowerBackwardsTwoKernel, [Float32, Float64])

# :nodoc:
class Num::ExpBackwardsKernel(T) < Num::Kernel(T)
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

          C[c] = A[a] * exp(B[b]);
        }
    }
    "
  end

  def call(gradient : Tensor(T, OCL(T)), a : Tensor(T, OCL(T)))
    result = gradient.class.new(gradient.shape)
    Cl.args(
      @kernel, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      gradient.data.shape, gradient.data.strides, gradient.offset, gradient.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, result.size)
    result
  end
end

create_kernel_children(ExpBackwardsKernel, [Float32, Float64])

# :nodoc:
class Num::TrigBackwardKernel(T) < Num::Kernel(T)
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
class Num::SinBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

        C[c] = A[a] * cos(B[b]);
      }
    }
    "
  end
end

create_kernel_children(SinBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::CosBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

            C[c] = A[a] * -sin(B[b]);
          }
        }
    "
  end
end

create_kernel_children(CosBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::TanBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

            C[c] = A[a] / pow(cos(B[b]), 2);
          }
        }
    "
  end
end

create_kernel_children(TanBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::AsinBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

            C[c] = fabs(B[b]) != 1 ? A[a] / sqrt((#{dtype})1 - pow(B[b], 2)) : (#{dtype})0 / (#{dtype})0;
          }
        }
    "
  end
end

create_kernel_children(AsinBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::AcosBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

            C[c] = fabs(B[b]) != 1 ? -A[a] / sqrt((#{dtype})1 - pow(B[b], 2)) : (#{dtype})0 / (#{dtype})0;
          }
        }
    "
  end
end

create_kernel_children(AcosBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::AtanBackwardKernel(T) < Num::TrigBackwardKernel(T)
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

            C[c] = A[a] / ((#{dtype})1 + pow(B[b], 2));
          }
        }
    "
  end
end

create_kernel_children(AtanBackwardKernel, [Float32, Float64])

# :nodoc:
class Num::AssignmentKernel(T) < Num::Kernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict A_shape,
        __global const int * restrict A_strides,
        const int A_offset,
        __global #{dtype} * restrict A,
        __global const int * restrict B_shape,
        __global const int * restrict B_strides,
        const int B_offset,
        __global const #{dtype} * restrict const B) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int b = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        A[a] = B[b];
      }
    }
    "
  end

  def call(a : Tensor(T, OCL(T)), b : Tensor(T, OCL(T)))
    b = b.broadcast_to(a.shape)
    Cl.args(
      @kernel,
      a.rank, a.size,
      a.data.shape, a.data.strides, a.offset, a.to_unsafe,
      b.data.shape, b.data.strides, b.offset, b.to_unsafe,
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, a.size)
    nil
  end
end

create_kernel_children(AssignmentKernel, [Int32, UInt32, Float32, Float64])

# :nodoc:
class Num::AssignmentScalarKernel(T) < Num::Kernel(T)
  def get_program(dtype)
    "
    #{super}

    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{@@name}(const int rank,
        const int len,
        __global const int * restrict A_shape,
        __global const int * restrict A_strides,
        const int A_offset,
        __global #{dtype} * restrict A,
        const #{dtype} B) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        const int a = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        A[a] = B;
      }
    }
    "
  end

  def call(a : Tensor(T, OCL(T)), b : T)
    Cl.args(
      @kernel,
      a.rank, a.size,
      a.data.shape, a.data.strides, a.offset, a.to_unsafe,
      b
    )
    Cl.run(Num::ClContext.instance.queue, @kernel, a.size)
    nil
  end
end

create_kernel_children(AssignmentScalarKernel, [Int32, UInt32, Float32, Float64])
