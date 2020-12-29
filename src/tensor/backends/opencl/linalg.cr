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

struct OCL(T) < Num::Backend::Storage(T)
  macro blast(name, *args, prefix = "")
    {%
      if T == Float32
        typ = :S.id
      elsif T == Float64
        typ = :D.id
      end
    %}
    event = Pointer(Void).malloc(1).unsafe_as(LibCL::ClEvent)
    queue = Num::ClContext.instance.queue
    LibBlast.clblast_{{prefix.id}}{{typ}}{{name}}({{*args}}, pointerof(queue), pointerof(event))
    Cl.check LibCL.cl_wait_for_events(1, pointerof(event))
    Cl.check LibCL.cl_release_event(event)
  end

  def blas_scale!(a : Number)
    blast(scal, @size, T.new(a), @data, 0, 1)
  end

  def blas_copy(other : OCL(T))
    blast(copy, @size, @data, 0, 1, other.data, 0, 1)
  end

  def blas_dot(other : OCL(T))
    scalar = Tensor(T).new([1], device: OCL(T))
    storage = scalar.storage
    blast(dot, @size, storage.data, 0, @data, @offset, 1, other.data, other.offset, 1)
    scalar
  end
end
