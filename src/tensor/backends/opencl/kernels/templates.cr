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

  # :nodoc:
  def custom_kernel(name, dtype, untyped, *args)
    const = "
    #{index_of_element}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    "

    variables = args.map_with_index do |arg, i|
      if i == 0
        "const int rank,
        const int len,
        __global const int * restrict #{arg}_shape,
        __global const int * restrict #{arg}_strides,
        const int #{arg}_offset,
        __global float * restrict #{arg},"
      else
        "__global const int * restrict #{arg}_shape,
        __global const int * restrict #{arg}_strides,
        const int #{arg}_offset,
        __global const float * restrict const #{arg},"
      end
    end

    indices = args.map do |arg|
      "const int #{arg.downcase} = opencl_getIndexOfElementID(rank, #{arg}_shape, #{arg}_strides, #{arg}_offset, elemID);"
    end

    fn = "
    #{index_of_element}
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void #{name}(#{variables.join("\n")[...-1]}) {
      for (int elemID = get_global_id(0); elemID < len; elemID += get_global_size(0)) {
        #{indices.join("\n")}

        #{untyped}
      }
    }
    "

    program = Cl.create_and_build(
      Num::ClContext.instance.context,
      fn, Num::ClContext.instance.device
    )
    {% if flag?(:debugcl) %}
      puts Cl.build_errors(program, [Num::ClContext.instance.device])
    {% end %}
    prok = Cl.create_kernel(program, name)
  end
end
