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

require "../cl_tensor"

# :nodoc:
module Num::Internal
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

  # :nodoc:
  macro compile_op(fn, suffix)
    # :nodoc:
    def compile_{{suffix.id}}(kern_name : String, ctype : String, op : String)
      cl_kernel = {{fn.id}}(kern_name, ctype, op)
      program = Cl.create_and_build(
        Num::ClContext.instance.context,
        cl_kernel, Num::ClContext.instance.device
      )
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
              Num::Internal.compile_{{fn.id}}({{arg[1]}}, {{dt[1]}}, {{arg[2]}})
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

  macro op(fn)
    def {{fn.id}}(a : ClTensor(Float32), b : ClTensor(Float32))
      prok = Num::Internal::ClCache.s{{fn.id}}_ew
      same_shape(a, b)
      t = ClTensor(Float32).new(a.shape)
      Cl.args(prok, a.to_unsafe, b.to_unsafe, t.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : ClTensor(Float32), b : ClTensor(Float32))
      prok = Num::Internal::ClCache.s{{fn.id}}_ew_inpl
      same_shape(a, b)
      Cl.args(prok, a.to_unsafe, b.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, a.size)
    end
  end

  op add
  op subtract
  op multiply
  op divide

  macro builtin(fn)
    def {{fn.id}}(a : ClTensor(Float32))
      prok = Num::Internal::ClCache.s{{fn.id}}_ew_fn
      t = ClTensor(Float32).new(a.shape)
      Cl.args(prok, a.to_unsafe, t.to_unsafe)
      Cl.run(Num::ClContext.instance.queue, prok, t.size)
      t
    end

    def {{fn.id}}!(a : ClTensor(Float32))
      prok = Num::Internal::ClCache.s{{fn.id}}_ew_fn_inpl
      Cl.args(prok, a.to_unsafe)
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

  private def same_shape(a : ClTensor, b : ClTensor)
    unless a.shape == b.shape
      raise Exception.new
    end
  end
end
