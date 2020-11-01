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

class Num::NN::ConvolutionalLayer(T) < Num::NN::Layer(T)
  getter in_shape : Array(Int32)
  getter weights : Num::Grad::Variable(T)
  getter bias : Num::Grad::Variable(T)
  getter padding : Tuple(Int32, Int32)
  getter stride : Tuple(Int32, Int32) = {1, 1}

  def initialize(
    context : Num::Grad::Context(T),
    in_shape : Array(Int),
    num_filters : Int,
    kernel_height : Int,
    kernel_width : Int,
    @padding = {0, 0},
    @stride = {1, 1}
  )
    @in_shape = in_shape.map &.to_i
    c_in, h_in, w_in = in_shape
    w = Num::NN.kaiming_normal(num_filters, c_in, kernel_height, kernel_width, dtype: T)
    b = T.zeros([num_filters, 1, 1])
    @weights = context.variable(w)
    @bias = context.variable(b)
  end

  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    output = \
       {% if flag?(:nnpack) %}
         Num::NN.conv2d(input.value, @weights.value, @bias.value, padding, stride)
       {% elsif flag?(:im2col) %}
         Num::NN.im2colgemm_conv2d(input.value, @weights.value, @bias.value, padding, stride)
       {% else %}
         Num::NN.im2colgemm_conv2d(input.value, @weights.value, @bias.value, padding, stride)
       {% end %}

    result = input.context.variable(output)

    if input.is_grad_needed || @weights.is_grad_needed || @bias.is_grad_needed
      gate = Num::NN::ConvolutionGate.new(input, @weights, @bias, @padding, @stride)
      gate.cache(result, input, @weights, @bias, @padding, @stride)
    end
    result
  end

  def variables : Array(Num::Grad::Variable(T))
    [weights, bias]
  end

  def output_shape : Array(Int32)
    kh = @weights.value.shape[2]
    kw = @weights.value.shape[3]
    ph = @padding[0]
    pw = @padding[1]
    sh = @stride[0]
    sw = @stride[1]

    ih = @in_shape[1]
    iw = @in_shape[2]
    dh = 1
    dw = 1

    r0 = @weights.value.shape[0]
    r1 = 1 + (ih + 2 * ph - (((kh - 1) * dh) + 1)) // sh
    r2 = 1 + (iw + 2 * pw - (((kw - 1) * dw) + 1)) // sw
    [r0, r1, r2]
  end
end
