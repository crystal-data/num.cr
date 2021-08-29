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

  class OpenCLKernelCache
    macro ops(*args)
      {% for dt in [{:s, "float"}, {:d, "double"}] %}
        {% for arg in args %}
          {% for fn in [:ew, :ew_inpl, :ew_fn, :ew_fn_inpl] %}
            class_getter {{dt[0].id}}{{arg[0]}}_{{fn.id}} : LibCL::ClProgram do
              Num.compile_{{fn.id}}({{arg[1] + "cached"}}, {{dt[1]}}, {{arg[2]}})
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
      {acos, "acos", "acos"},
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

    class_getter spower_ew do
      Num.custom_kernel("powerFloat", "float", "D[d] = pow(A[a], B[b]);", "D", "A", "B")
    end

    class_getter dpower_ew do
      Num.custom_kernel("powerDouble", "double", "D[d] = pow(A[a], B[b]);", "D", "A", "B")
    end

    class_getter spower_ew_inpl do
      Num.custom_kernel("powerFloatInpl", "float", "A[a] = pow(A[a], B[b]);", "A", "B")
    end

    class_getter dpower_ew_inpl do
      Num.custom_kernel("powerDoubleInpl", "double", "A[a] = pow(A[a], B[b]);", "A", "B")
    end

    class_getter divideBackwardsTwoFloat do
      Num.custom_kernel("divideBackwards", "float", "D[d] = -A[a] * B[b] / pow(C[c], 2);", "D", "A", "B", "C")
    end

    class_getter powerBackwardsOneFloat do
      Num.custom_kernel("powerBackwardsOne", "float", "D[d] = A[a] * C[c] * pow(B[b], C[c] == 0 ? (float)1 : C[c] - 1);", "D", "A", "B", "C")
    end

    class_getter powerBackwardsTwoFloat do
      Num.custom_kernel("powerBackwardsTwo", "float", "D[d] = A[a] * pow(B[b], C[c]) * log(B[b] == 0 ? (float)1 : B[b]);", "D", "A", "B", "C")
    end

    class_getter transposeFloat do
      Num.transpose_kernel
    end

    class_getter expBackwards do
      Num.custom_kernel("expBackwards", "float", "C[c] = A[a] * exp(B[b]);", "C", "A", "B")
    end

    class_getter sinBackwards do
      Num.custom_kernel("sinBackwards", "float", "C[c] = A[a] * cos(B[b]);", "C", "A", "B")
    end

    class_getter cosBackwards do
      Num.custom_kernel("sinBackwards", "float", "C[c] = A[a] * -sin(B[b]);", "C", "A", "B")
    end

    class_getter tanBackwards do
      Num.custom_kernel("tanBackwards", "float", "C[c] = A[a] / pow(cos(B[b]), 2);", "C", "A", "B")
    end

    class_getter asinBackwards do
      Num.custom_kernel(
        "tanBackwards",
        "float",
        "C[c] = fabs(B[b]) != 1 ? A[a] / sqrt((float)1 - pow(B[b], 2)) : (float)0 / (float)0;", "C", "A", "B"
      )
    end

    class_getter acosBackwards do
      Num.custom_kernel(
        "tanBackwards",
        "float",
        "C[c] = fabs(B[b]) != 1 ? -A[a] / sqrt((float)1 - pow(B[b], 2)) : (float)0 / (float)0;", "C", "A", "B"
      )
    end

    class_getter atanBackwards do
      Num.custom_kernel("tanBackwards", "float", "C[c] = A[a] / ((float)1 + pow(B[b], 2));", "C", "A", "B")
    end
  end
end
