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

require "../cltensor/cltensor"
require "../cltensor/creation"
require "../num/cl_builtin"

module Num
  def kernel_source(define : String? = nil)
    if !define.nil?
      define = "#define #{define} 1"
    end
    "
    // expected defines:
    // one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU ]
    #{define}

    #ifdef TANH
        #define ACTIVATION_FUNCTION(output) (tanh(output))
    #elif defined SCALEDTANH
        #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
    #elif SIGMOID
        #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
    #elif defined RELU
        #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
    #elif defined ELU
        #define ACTIVATION_FUNCTION(output) (output> 0 ? output : exp(output) - 1)
    #elif defined LINEAR
        #define ACTIVATION_FUNCTION(output) (output)
    #endif

    #ifdef ACTIVATION_FUNCTION // protect against not defined
      __kernel void activate(__global float *inout) {
          int gid = get_global_id(0);
          inout[gid] = ACTIVATION_FUNCTION(inout[gid]);
      }
    #endif

    #ifdef ACTIVATION_FUNCTION // protect against not defined
      __kernel void forwardNaive(__global const float *in, __global float *out) {
          int gid = get_global_id(0);
          out[gid] = ACTIVATION_FUNCTION(in[gid]);
      }
    #endif
    "
  end

  def activate(cl : ClTensor(Float32), name : String)
    cl_proc = compile_activation name
    Cl.args(cl_proc, cl.to_unsafe)
    Cl.run(Num::ClContext.instance.queue, cl_proc, cl.size)
    cl
  end

  def compile_activation(name : String)
    cl_kernel = kernel_source name
    program = Cl.create_and_build(Num::ClContext.instance.context, cl_kernel, Num::ClContext.instance.device)
    cl_proc = Cl.create_kernel(program, "activate")
  end

  def run_activation(cl : ClTensor(Float32), prok)
    Cl.args(prok, cl.to_unsafe)
    Cl.run(Num::ClContext.instance.queue, prok, cl.size)
    cl
  end

  def sigmoid(cl : ClTensor(Float32))
    activate cl, "SIGMOID"
  end

  def tanh(cl : ClTensor(Float32))
    activate cl, "TANH"
  end

  def tanh_scaled(cl : ClTensor(Float32))
    activate cl, "SCALEDTANH"
  end

  def relu(cl : ClTensor(Float32))
    activate cl, "RELU"
  end

  def elu(cl : ClTensor(Float32))
    activate cl, "ELU"
  end

  def linear(cl : ClTensor(Float32))
    activate cl, "LINEAR"
  end
end
