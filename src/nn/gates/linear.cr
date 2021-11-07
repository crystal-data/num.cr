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

# :nodoc:
class Num::NN::LinearGate(T) < Num::Grad::Gate(T)
  getter input : Num::Grad::Variable(T)
  getter weight : Num::Grad::Variable(T)
  getter bias : Num::Grad::Variable(T)

  def initialize(@input : Num::Grad::Variable(T), @weight : Num::Grad::Variable(T), @bias : Num::Grad::Variable(T))
  end

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    grad = payload.variable.grad

    result = [
      grad,
      grad,
      grad,
    ]

    if @input.requires_grad
      result[0] = grad.matmul(@weight.value)
    end

    if @weight.requires_grad
      result[1] = grad.transpose.matmul(@input.value)
    end

    if @bias.requires_grad
      result[2] = grad.sum(axis: 0)
    end

    result
  end

  def cache(result : Num::Grad::Variable(T), *args)
    input, weight, bias = args
    result.grad = T.zeros_like(result.value)
    Num::Grad.register("Linear", self, result, input, weight, bias)
  end
end
