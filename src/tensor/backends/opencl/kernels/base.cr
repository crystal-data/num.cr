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
  CTYPE_MAPPING = {
    Float32 => "float",
    Float64 => "double",
    Int32   => "int",
    UInt32  => "uint",
  }
end

# :nodoc:
abstract class Num::Kernel(T)
  macro inherited
    class_getter instance : self { self.new }
    @@name : String = ""
  end

  def initialize
    dtype = CTYPE_MAPPING[T]
    program_string = get_program(dtype)

    @program = Cl.create_and_build(
      Num::ClContext.instance.context,
      program_string,
      Num::ClContext.instance.device
    )

    {% if flag?(:debugcl) %}
      puts Cl.build_errors(@program, [Num::ClContext.instance.device])
    {% end %}

    @kernel = Cl.create_kernel(@program, @@name)
  end

  def get_program(dtype : String) : String
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

  def finalize
    Cl.release_program(@program)
    Cl.release_kernel(@kernel)
  end
end

# :nodoc:
macro create_kernel_children(kernel, dtypes)
  {% for dtype in dtypes %}
    # :nodoc:
    class Num::{{ dtype }}{{ kernel }} < Num::{{ kernel }}({{ dtype }})
      @@name = "{{ kernel }}"
    end
  {% end %}
end

# :nodoc:
macro call_opencl_kernel(generic, kernel, dtypes, *args)
  {% for dtype, index in dtypes %}
    {% if index == 0 %}
      \{% if {{ generic }} == {{ dtype }} %}
        Num::{{dtype}}{{ kernel }}.instance.call({{ *args }})
    {% else %}
      \{% elsif {{ generic }} == {{ dtype }} %}
        Num::{{ dtype }}{{ kernel }}.instance.call({{ *args }})
    {% end %}
    {% if index == dtypes.size - 1 %}
      \{% else %}
        \{% raise "Invalid dtype" %}
      \{% end %}
    {% end %}
  {% end %}
end
