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

class Num::NN::LinearLayer(T) < Num::NN::Layer(T)
  getter weights : Num::Grad::Variable(T)
  getter bias : Num::Grad::Variable(T)

  def initialize(context : Num::Grad::Context(T), inp_dim : Int, outp_dim : Int)
    w = T.rand([outp_dim, inp_dim])
    b = T.rand([1, outp_dim])
    @weights = context.variable(w)
    @bias = context.variable(b)
  end

  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    output = input.value.matmul(@weights.value.transpose) + bias.value
    result = input.context.variable(output)

    if input.is_grad_needed || @weights.is_grad_needed || @bias.is_grad_needed
      gate = Num::NN::LinearGate.new(input, @weights, @bias)
      gate.cache(result, input, @weights, @bias)
    end
    result
  end

  def variables : Array(Num::Grad::Variable(T))
    [weights, bias]
  end
end
