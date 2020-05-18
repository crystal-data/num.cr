# Copyright (c) 2020 Crystal Data Contributors
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

require "../cltensor"

module Num
  extend self

  private def gen_cl_apply3(kern_name : String, ctype : String, op : String) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global const #{ctype} *b, __global #{ctype} *c) {
        int gid = get_global_id(0);
        c[gid] = a[gid] #{op} b[gid];
    }
    "
  end

  private def gen_cl_apply3_inpl(kern_name : String, ctype : String, op : String) : String
    "
    __kernel void #{kern_name}(__global #{ctype} *a, __global const #{ctype} *b) {
        int gid = get_global_id(0);
        a[gid] = a[gid] #{op} b[gid];
    }
    "
  end

  private def gen_cl_apply2(kern_name : String, ctype : String, op : String, rhs : Number) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global #{ctype} *b) {
        int gid = get_global_id(0);
        b[gid] = a[gid] #{op} #{rhs};
    }
    "
  end

  private def gen_cl_apply2_inpl(kern_name : String, ctype : String, op : String, rhs : Number) : String
    "
    __kernel void #{kern_name}(__global #{ctype} *a) {
        int gid = get_global_id(0);
        a[gid] = a[gid] #{op} #{rhs};
    }
    "
  end

  private def gen_cl_apply2_lhs(kern_name : String, ctype : String, op : String, lhs : Number) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global #{ctype} *b) {
        int gid = get_global_id(0);
        b[gid] = #{lhs} #{op} a[gid];
    }
    "
  end

  private def gen_cl_apply2_lhs_inpl(kern_name : String, ctype : String, op : String, lhs : Number) : String
    "
    __kernel void #{kern_name}(__global #{ctype} *a) {
        int gid = get_global_id(0);
        a[gid] = #{lhs} #{op} a[gid]
    }
    "
  end

  private def gen_cl_math_fn1(kern_name : String, ctype : String, fn : String) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global #{ctype} *b) {
        int gid = get_global_id(0);
        b[gid] = #{fn}(a[gid]);
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

  macro gen_cl_infix_op(dtype, ctype, fn, cname, op)
    def {{fn.id}}(a : ClTensor({{dtype}}), b : ClTensor({{dtype}}))
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_apply3({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})

      Cl.args(cl_proc, a.to_unsafe, b.to_unsafe, result.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, result.size)
      result
    end

    def {{fn.id}}!(a : ClTensor({{dtype}}), b : ClTensor({{dtype}}))
      cl_kernel = gen_cl_apply3_inpl({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})

      Cl.args(cl_proc, a.to_unsafe, b.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, a.size)
    end

    def {{fn.id}}(a : ClTensor({{dtype}}), b : {{dtype}})
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_apply2({{cname}}, {{ctype}}, {{op}}, b)
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})
      Cl.args(cl_proc, a.to_unsafe, result.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, result.size)
      result
    end

    def {{fn.id}}!(a : ClTensor({{dtype}}), b : {{dtype}})
      cl_kernel = gen_cl_apply2_inpl({{cname}}, {{ctype}}, {{op}}, b)
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})
      Cl.args(cl_proc, a.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, a.size)
    end

    def {{fn.id}}(b : {{dtype}}, a : ClTensor({{dtype}}))
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_apply2_lhs({{cname}}, {{ctype}}, {{op}}, b)
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})
      Cl.args(cl_proc, a.to_unsafe, result.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, result.size)
      result
    end
  end

  macro gen_cl_math_fn_op(dtype, ctype, fn, cname, op)
    def {{fn.id}}(a : ClTensor({{dtype}}))
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_math_fn1({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)
      cl_proc = Cl.create_kernel(program, {{cname}})

      Cl.args(cl_proc, a.to_unsafe, result.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, result.size)
      result
    end

    def {{fn.id}}!(a : ClTensor({{dtype}}))
      cl_kernel = gen_cl_math_fn1_inpl({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)
      cl_proc = Cl.create_kernel(program, {{cname}})

      Cl.args(cl_proc, a.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, cl_proc, a.size)
    end
  end

  gen_cl_infix_op(Float64, "double", "add", "add_vector", "+")
  gen_cl_infix_op(Float32, "float", "add", "add_vector", "+")

  gen_cl_infix_op(Float64, "double", "subtract", "subtract_vector", "-")
  gen_cl_infix_op(Float32, "float", "subtract", "subtract_vector", "-")

  gen_cl_infix_op(Float64, "double", "multiply", "multiply_vector", "*")
  gen_cl_infix_op(Float32, "float", "multiply", "multiply_vector", "*")

  gen_cl_infix_op(Float64, "double", "divide", "divide_vector", "/")
  gen_cl_infix_op(Float32, "float", "divide", "divide_vector", "/")

  gen_cl_math_fn_op(Float64, "double", "acos", "acos_vector", "acos")
  gen_cl_math_fn_op(Float32, "float", "acos", "acos_vector", "acos")

  gen_cl_math_fn_op(Float64, "double", "acosh", "acosh_vector", "acosh")
  gen_cl_math_fn_op(Float32, "float", "acosh", "acosh_vector", "acosh")

  gen_cl_math_fn_op(Float64, "double", "acospi", "acospi_vector", "acospi")
  gen_cl_math_fn_op(Float32, "float", "acospi", "acospi_vector", "acospi")

  gen_cl_math_fn_op(Float64, "double", "asin", "asin_vector", "asin")
  gen_cl_math_fn_op(Float32, "float", "asin", "asin_vector", "asin")

  gen_cl_math_fn_op(Float64, "double", "asinh", "asinh_vector", "asinh")
  gen_cl_math_fn_op(Float32, "float", "asinh", "asinh_vector", "asinh")

  gen_cl_math_fn_op(Float64, "double", "asinpi", "asinpi_vector", "asinpi")
  gen_cl_math_fn_op(Float32, "float", "asinpi", "asinpi_vector", "asinpi")

  gen_cl_math_fn_op(Float64, "double", "atan", "atan_vector", "atan")
  gen_cl_math_fn_op(Float32, "float", "atan", "atan_vector", "atan")

  gen_cl_math_fn_op(Float64, "double", "atanh", "atanh_vector", "atanh")
  gen_cl_math_fn_op(Float32, "float", "atanh", "atanh_vector", "atanh")

  gen_cl_math_fn_op(Float64, "double", "atanpi", "atanpi_vector", "atanpi")
  gen_cl_math_fn_op(Float32, "float", "atanpi", "atanpi_vector", "atanpi")

  gen_cl_math_fn_op(Float64, "double", "cbrt", "cbrt_vector", "cbrt")
  gen_cl_math_fn_op(Float32, "float", "cbrt", "cbrt_vector", "cbrt")

  gen_cl_math_fn_op(Float64, "double", "ceil", "ceil_vector", "ceil")
  gen_cl_math_fn_op(Float32, "float", "ceil", "ceil_vector", "ceil")

  gen_cl_math_fn_op(Float64, "double", "cos", "cos_vector", "cos")
  gen_cl_math_fn_op(Float32, "float", "cos", "cos_vector", "cos")

  gen_cl_math_fn_op(Float64, "double", "cosh", "cosh_vector", "cosh")
  gen_cl_math_fn_op(Float32, "float", "cosh", "cosh_vector", "cosh")

  gen_cl_math_fn_op(Float64, "double", "cospi", "cospi_vector", "cospi")
  gen_cl_math_fn_op(Float32, "float", "cospi", "cospi_vector", "cospi")

  gen_cl_math_fn_op(Float64, "double", "erfc", "erfc_vector", "erfc")
  gen_cl_math_fn_op(Float32, "float", "erfc", "erfc_vector", "erfc")

  gen_cl_math_fn_op(Float64, "double", "erf", "erf_vector", "erf")
  gen_cl_math_fn_op(Float32, "float", "erf", "erf_vector", "erf")

  gen_cl_math_fn_op(Float64, "double", "exp", "exp_vector", "exp")
  gen_cl_math_fn_op(Float32, "float", "exp", "exp_vector", "exp")

  gen_cl_math_fn_op(Float64, "double", "exp2", "exp2_vector", "exp2")
  gen_cl_math_fn_op(Float32, "float", "exp2", "exp2_vector", "exp2")

  gen_cl_math_fn_op(Float64, "double", "exp10", "exp10_vector", "exp10")
  gen_cl_math_fn_op(Float32, "float", "exp10", "exp10_vector", "exp10")

  gen_cl_math_fn_op(Float64, "double", "expm1", "expm1_vector", "expm1")
  gen_cl_math_fn_op(Float32, "float", "expm1", "expm1_vector", "expm1")

  gen_cl_math_fn_op(Float64, "double", "fabs", "fabs_vector", "fabs")
  gen_cl_math_fn_op(Float32, "float", "fabs", "fabs_vector", "fabs")

  gen_cl_math_fn_op(Float64, "double", "floor", "floor_vector", "floor")
  gen_cl_math_fn_op(Float32, "float", "floor", "floor_vector", "floor")

  gen_cl_math_fn_op(Float64, "double", "lgamma", "lgamma_vector", "lgamma")
  gen_cl_math_fn_op(Float32, "float", "lgamma", "lgamma_vector", "lgamma")

  gen_cl_math_fn_op(Float64, "double", "log", "log_vector", "log")
  gen_cl_math_fn_op(Float32, "float", "log", "log_vector", "log")

  gen_cl_math_fn_op(Float64, "double", "log2", "log2_vector", "log2")
  gen_cl_math_fn_op(Float32, "float", "log2", "log2_vector", "log2")

  gen_cl_math_fn_op(Float64, "double", "log10", "log10_vector", "log10")
  gen_cl_math_fn_op(Float32, "float", "log10", "log10_vector", "log10")

  gen_cl_math_fn_op(Float64, "double", "log1p", "log1p_vector", "log1p")
  gen_cl_math_fn_op(Float32, "float", "log1p", "log1p_vector", "log1p")

  gen_cl_math_fn_op(Float64, "double", "logb", "logb_vector", "logb")
  gen_cl_math_fn_op(Float32, "float", "logb", "logb_vector", "logb")

  gen_cl_math_fn_op(Float64, "double", "rint", "rint_vector", "rint")
  gen_cl_math_fn_op(Float32, "float", "rint", "rint_vector", "rint")

  gen_cl_math_fn_op(Float64, "double", "round", "round_vector", "round")
  gen_cl_math_fn_op(Float32, "float", "round", "round_vector", "round")

  gen_cl_math_fn_op(Float64, "double", "sqrt", "sqrt_vector", "sqrt")
  gen_cl_math_fn_op(Float32, "float", "sqrt", "sqrt_vector", "sqrt")

  gen_cl_math_fn_op(Float64, "double", "rsqrt", "rsqrt_vector", "rsqrt")
  gen_cl_math_fn_op(Float32, "float", "rsqrt", "rsqrt_vector", "rsqrt")

  gen_cl_math_fn_op(Float64, "double", "sin", "sin_vector", "sin")
  gen_cl_math_fn_op(Float32, "float", "sin", "sin_vector", "sin")

  gen_cl_math_fn_op(Float64, "double", "sinh", "sinh_vector", "sinh")
  gen_cl_math_fn_op(Float32, "float", "sinh", "sinh_vector", "sinh")

  gen_cl_math_fn_op(Float64, "double", "sinpi", "sinpi_vector", "sinpi")
  gen_cl_math_fn_op(Float32, "float", "sinpi", "sinpi_vector", "sinpi")

  gen_cl_math_fn_op(Float64, "double", "sqrt", "sqrt_vector", "sqrt")
  gen_cl_math_fn_op(Float32, "float", "sqrt", "sqrt_vector", "sqrt")

  gen_cl_math_fn_op(Float64, "double", "tan", "tan_vector", "tan")
  gen_cl_math_fn_op(Float32, "float", "tan", "tan_vector", "tan")

  gen_cl_math_fn_op(Float64, "double", "tanh", "tanh_vector", "tanh")
  gen_cl_math_fn_op(Float32, "float", "tanh", "tanh_vector", "tanh")

  gen_cl_math_fn_op(Float64, "double", "tanpi", "tanpi_vector", "tanpi")
  gen_cl_math_fn_op(Float32, "float", "tanpi", "tanpi_vector", "tanpi")

  gen_cl_math_fn_op(Float64, "double", "tgamma", "tgamma_vector", "tgamma")
  gen_cl_math_fn_op(Float32, "float", "tgamma", "tgamma_vector", "tgamma")

  gen_cl_math_fn_op(Float64, "double", "trunc", "trunc_vector", "trunc")
  gen_cl_math_fn_op(Float32, "float", "trunc", "trunc_vector", "trunc")
end
