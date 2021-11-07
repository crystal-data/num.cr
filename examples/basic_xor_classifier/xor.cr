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

require "../../src/num"

Num::Rand.set_seed(2)

ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

bsz = 32

x_train_bool = Tensor.random(0_u8...2_u8, [bsz * 100, 2])

y_bool = x_train_bool[..., ...1] ^ x_train_bool[..., 1...]

x_train = ctx.variable(x_train_bool.as_type(Float64))
y = y_bool.as_type(Float64)

net = Num::NN::Network.new(ctx) do
  input [2]
  linear 3
  relu
  linear 1
  sgd 0.7
  sigmoid_cross_entropy_loss
end

losses = [] of Float64

50.times do |epoch|
  100.times do |batch_id|
    offset = batch_id * 32
    x = x_train[offset...offset + 32]
    target = y[offset...offset + 32]

    y_pred = net.forward(x)

    loss = net.loss(y_pred, target)

    puts "Epoch is: #{epoch}"
    puts "Batch id: #{batch_id}"
    puts "Loss is: #{loss.value.value}"
    losses << loss.value.value

    loss.backprop
    net.optimizer.update
  end
end
