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

ctx = Num::Grad::Context(Tensor(Float32)).new

dataset = Num::NN.load_mnist_dataset

x_train = dataset.features / 255_f32
x_train = ctx.variable(x_train)
y_train = dataset.labels

net = Num::NN::Network.new(ctx) do
  linear 784, 32
  relu
  linear 32, 10
  softmax_cross_entropy_loss
  sgd 0.01
end

batch_size = 32

accuracies = [] of Float64

100.times do |epoch|
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

  accuracy = y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size
  accuracies << accuracy

  puts "Epoch: #{epoch} | Accuracy: #{accuracy}"
end

x_test = ctx.variable(dataset.test_features)
y_test = dataset.test_labels

a = net.forward(x_test).value.argmax(1)
b = y_test.argmax(1)

result = a.map(b) do |i, j|
  i == j ? 1 : 0
end

puts "Test accuracy: #{result.mean}"

Num::Plot::Plot.plot do
  scatter (0...accuracies.size), accuracies
  x_label "Epochs"
  y_label "Accuracy"
  label "Crappy MNIST Accuracy"
end
