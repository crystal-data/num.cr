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

require "num"
require "ishi"

# Basic XOR input data

x_train = [
  [0, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1],
].to_tensor.as_type(Float64).transpose

y_train = [[0, 1, 1, 1, 1, 1, 1, 0]].to_tensor.as_type(Float64)

# An example of defining a custom layer to be used in
# a Network
class MyCustomActivation(T) < Num::NN::ActivationLayer(T)
  def activate(input : Tensor(T)) : Tensor(T)
    Num.max(T.new(0), input)
  end

  def derive(gradient : Tensor(T)) : Tensor(T)
    gradient.map do |i|
      i >= 0 ? T.new(1) : T.new(0)
    end
  end
end

m = x_train.shape[1]
epochs = 1500

options = {
  learning_rate: 0.1,
}

costs = [] of Float64

# Classes provided will be used instead of looking up via symbols
# to allow for custom, user-defined classes
net = Num::NN::Network(Float64).new(**options) do
  layer(3, 6, :tanh)
  layer(6, 6, MyCustomActivation)
  layer(6, 1, :sigmoid)
end

# Using the same standard log loss as the generic example
epochs.times do
  a = net.forward(x_train)
  cost = 1/m * Num.sum(Num::NN.logloss(y_train, a))
  costs << cost
  loss_gradient = Num::NN.d_logloss(y_train, a)
  net.backward(loss_gradient)
end

puts net.forward(x_train)

Ishi.new do
  plot((0...costs.size).to_a, costs)
end

# SAMPLE RUN:
