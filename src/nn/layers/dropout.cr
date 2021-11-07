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

# Dilution (also called Dropout) is a regularization technique for
# reducing overfitting in artificial neural networks by preventing
# complex co-adaptations on training data. It is an efficient way
# of performing model averaging with neural networks. The term dilution
# refers to the thinning of the weights. The term dropout refers to
# randomly "dropping out", or omitting, units (both hidden and visible)
# during the training process of a neural network. Both the thinning
# of weights and dropping out units trigger the same type of regularization,
# and often the term dropout is used when referring to the dilution of
# weights.
class Num::NN::DropoutLayer(T) < Num::NN::Layer(T)
  @prob : Float32 | Float64
  getter output_shape : Array(Int32)

  # Initialize a dropout layer in a `Num::NN::Network(T)`
  #
  # ## Arguments
  #
  # * context : `Num::Grad::Context(T)` - Context associated with the network,
  #   used only for determining generic type.
  # * output_shape : `Array(Int32)` - Cached output shape
  # * prob : `Float32` - Probability of dropping out a value when performing
  #   a forward pass
  def initialize(
    context : Num::Grad::Context(T),
    @output_shape : Array(Int32),
    prob = 0.5_f32
  )
    @prob = 1 - prob
  end

  # Computes the forward pass of a `Num::NN::Network`.  This will remove
  # a certain amount of neurons from the input variable, and scale the
  # remaining values by the probability of removal.
  #
  # ## Arguments
  #
  # * input : `Num::Grad::Variable(T)` - Input variable to the layer
  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    mask = T.binomial(input.value.shape, 1, @prob)
    output = Num::NN.dropout(input.value, mask, @prob)
    result = input.context.variable(output)

    if input.is_grad_needed
      gate = Num::NN::DropoutGate.new(mask, @prob)
      gate.cache(result, input)
    end
    result
  end
end
