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

struct NumInternal::ClInfo
  getter device : LibCL::ClDeviceId
  getter context : LibCL::ClContext
  getter queue : LibCL::ClCommandQueue

  def initialize(@device : LibCL::ClDeviceId, @context : LibCL::ClContext, @queue : LibCL::ClCommandQueue)
  end
end

class Num::ClContext
  {% if flag?(:opencl_any) %}
    class_getter instance : NumInternal::ClInfo { NumInternal::ClInfo.new(*Cl.single_device_defaults) }
  {% else %}
    class_getter instance : NumInternal::ClInfo { NumInternal::ClInfo.new(*Cl.first_gpu_defaults) }
  {% end %}

  def self.set_device(device)
    context = Cl.create_context([device])
    queue = Cl.command_queue_for(context, device)
    @@instance = NumInternal::ClInfo.new(device, context, queue)
  end
end
