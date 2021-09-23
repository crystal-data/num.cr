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

class Num::NN::MaxPoolGate(T) < Num::Grad::Gate(T)
  getter input_shape : Array(Int32)
  getter max_indices : Tensor(Int32)
  getter kernel : Tuple(Int32, Int32)
  getter padding : Tuple(Int32, Int32)
  getter stride : Tuple(Int32, Int32)

  def initialize(
    @input_shape : Array(Int32),
    @max_indices : Tensor(Int32),
    @kernel : Tuple(Int32, Int32),
    @padding : Tuple(Int32, Int32),
    @stride : Tuple(Int32, Int32)
  )
  end

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad

    r0 = Num::NN.maxpool_backward(
      @input_shape,
      @max_indices,
      gradient
    )

    [r0]
  end

  def cache(result : Num::Grad::Variable(T), *args)
    input, kernel, padding, stride = args

    result.grad = T.zeros_like(result.value)
    result.requires_grad = true

    Num::Grad.register("Maxpool", self, result, input)
  end
end
