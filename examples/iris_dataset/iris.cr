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

ctx = Num::Grad::Context(Tensor(Float64)).new

labels, x_train, y_train = Num::NN.load_iris_dataset

x_train = (x_train - x_train.mean(axis: 0)) / x_train.std(axis: 0)
x_train = ctx.variable(x_train)

net = Num::NN::Network.new(ctx) do
  linear 4, 3
  relu
  linear 3, 3
  sgd 0.9
  sigmoid_cross_entropy_loss
end

batch_size = 10

10.times do |epoch|
  y_trues = [] of Int32
  y_preds = [] of Int32

  (y_train.shape[0] // batch_size).times do |batch_id|
    offset = batch_id * batch_size
    x = x_train[offset...offset + batch_size]
    target = y_train[offset...offset + batch_size]

    output = net.forward(x)

    loss = net.loss(output, target)

    y_trues += target.argmax(axis: 1).to_a
    y_preds += output.value.argmax(axis: 1).to_a

    loss.backprop
    net.optimizer.update
  end

  puts "Epoch: #{epoch} | Accuracy: #{y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size}"
end
