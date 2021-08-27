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
  def transpose_kernel
    kernel = "
    __kernel void matrixTranspose(const int heightA, const int widthA,
    			__global
    	const float *a,
    		__global float *a_T)
    {

    	const int colA = get_global_id(0);

    	for (int rowA = 0; rowA < heightA; rowA++)
    	{
    		a_T[colA *heightA + rowA] = a[rowA *widthA + colA];
    	}
    }
    "
    program = Cl.create_and_build(
      Num::ClContext.instance.context,
      kernel, Num::ClContext.instance.device
    )
    {% if flag?(:debugcl) %}
      puts Cl.build_errors(program, [Num::ClContext.instance.device])
    {% end %}
    Cl.create_kernel(program, "matrixTranspose")
  end
end
