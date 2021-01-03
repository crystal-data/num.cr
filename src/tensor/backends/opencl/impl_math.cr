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
  private def index_of_element : String
    "
    int opencl_getIndexOfElementID(
      const int rank,
      __global const int * restrict const shape,
      __global const int * restrict const strides,
      const int offset,
      const int element_id) {
      int real_idx = offset;
      int currentOffset = element_id;
      int dimIdx = 0;
      for (int k = rank - 1; k >= 0; --k) {
        dimIdx = currentOffset % shape[k];
        currentOffset /= shape[k];
        real_idx += dimIdx * strides[k];
      }
      return real_idx;
    }
    "
  end

  private def gen_cl_apply3(kern_name : String, ctype : String, op : String) : String
    "
    #{index_of_element}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void #{kern_name}
                (const int rank,
                const int len,
                __global const int * restrict dst_shape,
                __global const int * restrict dst_strides,
                const int dst_offset,
                __global       #{ctype} * restrict const dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{ctype} * restrict const A_data,
                __global const int * restrict B_shape,
                __global const int * restrict B_strides,
                const int B_offset,
                __global const #{ctype} * restrict const B_data)
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        dst_data[dst_real_idx] = A_data[A_real_idx] #{op} B_data[B_real_idx];
      }
    }
    "
  end

  private def gen_cl_apply3_inpl(kern_name : String, ctype : String, op : String) : String
    "
    #{index_of_element}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void #{kern_name}
                (const int rank,
                const int len,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global #{ctype} * A_data,
                __global const int * restrict B_shape,
                __global const int * restrict B_strides,
                const int B_offset,
                __global const #{ctype} * restrict const B_data)
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);
        A_data[A_real_idx] = A_data[A_real_idx] #{op} B_data[B_real_idx];
      }
    }
    "
  end

  private def gen_cl_math_fn1(kern_name : String, ctype : String, fn : String) : String
    "
    #{index_of_element}

    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    __kernel void #{kern_name}
                (const int rank,
                const int len,
                __global const int * restrict dst_shape,
                __global const int * restrict dst_strides,
                const int dst_offset,
                __global #{ctype} * dst_data,
                __global const int * restrict A_shape,
                __global const int * restrict A_strides,
                const int A_offset,
                __global const #{ctype} * restrict const A_data)
    {
      for (int elemID = get_global_id(0);
      elemID < len;
      elemID += get_global_size(0)) {
        const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
        const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
        dst_data[dst_real_idx] = #{fn}(A_data[A_real_idx]);
      }
    }
    "
  end

  private def gen_cl_math_fn1_inpl(kern_name : String, ctype : String, fn : String) : String
    "
    __kernel void #{kern_name}(__global #{ctype} *a) {
        int gid = get_global_id(0);
        a[gid] = #{fn}(a[gid]);
    }
    "
  end

  private macro compile_op(fn, suffix)
    # :nodoc:
    def compile_{{suffix.id}}(kern_name : String, ctype : String, op : String)
      cl_kernel = {{fn.id}}(kern_name, ctype, op)
      program = Cl.create_and_build(
        Num::ClContext.instance.context,
        cl_kernel, Num::ClContext.instance.device
      )
      {% if flag?(:debugcl) %}
        puts Cl.build_errors(program, [Num::ClContext.instance.device])
      {% end %}
      Cl.create_kernel(program, kern_name)
    end
  end

  compile_op gen_cl_apply3, ew
  compile_op gen_cl_apply3_inpl, ew_inpl
  compile_op gen_cl_math_fn1, ew_fn
  compile_op gen_cl_math_fn1_inpl, ew_fn_inpl

  # :nodoc:
  class ClCache
    macro ops(*args)
      {% for dt in [{:s, "float"}, {:d, "double"}] %}
        {% for arg in args %}
          {% for fn in [:ew, :ew_inpl, :ew_fn, :ew_fn_inpl] %}
            class_getter {{dt[0].id}}{{arg[0]}}_{{fn.id}} : LibCL::ClProgram do
              Num.compile_{{fn.id}}({{arg[1]}}, {{dt[1]}}, {{arg[2]}})
            end
          {% end %}
        {% end %}
      {% end %}
    end

    ops(
      {add, "add", "+"},
      {subtract, "subtract", "-"},
      {multiply, "multiply", "*"},
      {divide, "divide", "/"},
      {acospi, "acospi", "acospi"},
      {asin, "asin", "asin"},
      {asinh, "asinh", "asinh"},
      {asinpi, "asinpi", "asinpi"},
      {atan, "atan", "atan"},
      {atanh, "atanh", "atanh"},
      {atanpi, "atanpi", "atanpi"},
      {cbrt, "cbrt", "cbrt"},
      {ceil, "ceil", "ceil"},
      {cos, "cos", "cos"},
      {cosh, "cosh", "cosh"},
      {cospi, "cospi", "cospi"},
      {erfc, "erfc", "erfc"},
      {erf, "erf", "erf"},
      {exp, "exp", "exp"},
      {exp2, "exp2", "exp2"},
      {exp10, "exp10", "exp10"},
      {expm1, "expm1", "expm1"},
      {fabs, "fabs", "fabs"},
      {floor, "floor", "floor"},
      {lgamma, "lgamma", "lgamma"},
      {log, "log", "log"},
      {log2, "log2", "log2"},
      {log10, "log10", "log10"},
      {log1p, "log1p", "log1p"},
      {logb, "logb", "logb"},
      {rint, "rint", "rint"},
      {round, "round", "round"},
      {sqrt, "sqrt", "sqrt"},
      {rsqrt, "rsqrt", "rsqrt"},
      {sin, "sin", "sin"},
      {sinh, "sinh", "sinh"},
      {sinpi, "sinpi", "sinpi"},
      {tan, "tan", "tan"},
      {tanh, "tanh", "tanh"},
      {tanpi, "tanpi", "tanpi"},
      {tgamma, "tgamma", "tgamma"},
      {trunc, "trunc", "trunc"},
    )
  end
end

module Num
  extend self

  macro elementwise_op(fn, dtype, prefix)
    def {{fn.id}}(a : Tensor({{dtype}}, OCL({{dtype}})), b : Tensor({{dtype}}, OCL({{dtype}})))
      prok = ClCache.{{prefix}}{{fn.id}}_ew
      a, b = a.broadcast(b)
      t = Tensor({{dtype}}, OCL({{dtype}})).new(a.shape)
      Cl.args(prok, t.rank, t.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe, b.data.shape, b.data.strides, b.offset, b.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor({{dtype}}, OCL({{dtype}})), b : Tensor({{dtype}}, OCL({{dtype}})))
      prok = ClCache.s{{fn.id}}_ew_inpl
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

  macro builtin(fn)
    def {{fn.id}}(a : Tensor(Float32, OCL(Float32)))
      prok = ClCache.s{{fn.id}}_ew_fn
      t = Tensor(Float32, OCL(Float32)).new(a.shape)
      Cl.args(prok, a.rank, a.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor(Float32, OCL(Float32)))
      prok = ClCache.s{{fn.id}}_ew_fn_inpl
      Cl.args(prok, a.rank, a.size, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok)
    end

    def {{fn.id}}(a : Tensor(Float64, OCL(Float64)))
      prok = ClCache.d{{fn.id}}_ew_fn
      t = Tensor(Float32, OCL(Float32)).new(a.shape)
      Cl.args(prok, a.rank, a.size, t.data.shape, t.data.strides, t.offset, t.data.to_unsafe, a.data.shape, a.data.strides, a.offset, a.data.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : Tensor(Float64, OCL(Float64)))
      prok = ClCache.d{{fn.id}}_ew_fn_inpl
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

  private def same_shape(a : Tensor, b : Tensor)
    unless a.shape == b.shape
      raise Exception.new
    end
  end
end
