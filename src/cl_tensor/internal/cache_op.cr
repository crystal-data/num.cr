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

  private def gen_cl_apply2_rhs(kern_name : String, ctype : String, op : String) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global #{ctype} *b, __global #{ctype} c) {
        int gid = get_global_id(0);
        b[gid] = a[gid] #{op} c;
    }
    "
  end

  private def gen_cl_apply2_rhs_inpl(kern_name : String, ctype : String, op : String) : String
    "
    __kernel void #{kern_name}(__global #{ctype} *a, __global #{ctype} b) {
        int gid = get_global_id(0);
        a[gid] = a[gid] #{op} b;
    }
    "
  end

  # :nodoc:
  macro compile(fn, suffix)
    # :nodoc:
    def compile_{{suffix.id}}(kern_name : String, ctype : String, op : String)
      cl_kernel = {{fn.id}}(kern_name, ctype, op)
      puts cl_kernel
      program = Cl.create_and_build(
        Num::ClContext.instance.context,
        cl_kernel, Num::ClContext.instance.device
      )
      Cl.create_kernel(program, kern_name)
    end
  end

  compile gen_cl_apply3, ew
  compile gen_cl_apply3_inpl, ew_inpl
  compile gen_cl_apply2_rhs, rhs
  compile gen_cl_apply2_rhs_inpl, rhs_inpl

  # :nodoc:
  class ClCache
    macro ops(*args)
      {% for dt in [{:s, "float"}, {:d, "double"}] %}
        {% for arg in args %}
          {% for fn in [:ew, :ew_inpl, :rhs, :rhs_inpl] %}
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
      {divide, "divide", "/"}
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

  private def same_shape(a : ClTensor, b : ClTensor)
    unless a.shape == b.shape
      raise Exception.new
    end
  end
end

a = ClTensor(Float32).new([3, 2, 2], 1.32_f32)
b = ClTensor(Float32).new([3, 2, 2], 1.48_f32)

puts Num.add(a, 3).cpu
