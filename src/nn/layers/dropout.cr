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

class Num::NN::DropoutLayer(T) < Num::NN::Layer(T)
  getter prob : Float32
  getter output_shape : Array(Int32)

  def initialize(context : Num::Grad::Context(T), @output_shape : Array(Int32), @prob = 0.5_f32)
  end

  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    mask = T.binomial(input.value.shape, 1, @prob) / @prob
    output = input.value * mask
    result = input.context.variable(output)

    if input.is_grad_needed
      gate = Num::NN::DropoutGate.new(mask)
      gate.cache(result, input)
    end
    result
  end
end
