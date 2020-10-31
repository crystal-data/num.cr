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

class Num::NN::MaxPoolLayer(T) < Num::NN::Layer(T)
  getter input_shape : Array(Int32)
  getter kernel : Tuple(Int32, Int32)
  getter padding : Tuple(Int32, Int32)
  getter stride : Tuple(Int32, Int32)

  def initialize(context : Num::Grad::Context(T), input_shape : Array(Int), kernel : Tuple(Int, Int), padding : Tuple(Int, Int), stride : Tuple(Int, Int))
    @input_shape = input_shape.map &.to_i
    @kernel = kernel.map &.to_i
    @padding = padding.map &.to_i
    @stride = stride.map &.to_i
  end

  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    max_indices, output = Num::NN.maxpool(input.value, @kernel, @padding, @stride)
    result = input.context.variable(output)

    if input.is_grad_needed
      gate = Num::NN::MaxPoolGate(T).new(input.value.shape, max_indices, @kernel, @padding, @stride)
      gate.cache(result, input, kernel, padding, stride)
    end
    result
  end

  def output_shape : Array(Int32)
    c, h, w = @input_shape
    kh = @kernel[0]
    kw = @kernel[1]
    ph = @padding[0]
    pw = @padding[1]
    sh = @stride[0]
    sw = @stride[1]

    r0 = c
    r1 = (h + (2 * ph) - kh) // sh + 1
    r2 = (w + (2 * pw) - kw) // sw + 1
    [r0, r1, r2]
  end
end
