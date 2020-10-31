require "../../src/num"

dataset = Num::NN.load_mnist_dataset
ctx = Num::Grad::Context(Tensor(Float32)).new

batch_size = 32

net = Num::NN::Network.new(ctx) do
  input [1, 28, 28]
  conv2d 20, 5, 5
  relu
  maxpool({2, 2}, {0, 0}, {2, 2})
  conv2d 20, 5, 5
  maxpool({2, 2}, {0, 0}, {2, 2})
  flatten
  linear 10
  relu
  linear 10
  softmax_cross_entropy_loss
  sgd 0.01
end

x_train = ctx.variable((dataset.features / 255_f32).reshape(-1, 1, 28, 28))
y_train = dataset.labels

losses = [] of Float32

5.times do |epoch|
  y_trues = [] of Int32
  y_preds = [] of Int32

  (x_train.value.shape[0] // batch_size).times do |batch_id|
    offset = batch_id * batch_size
    x = x_train[offset...offset + batch_size]
    target = y_train[offset...offset + batch_size]

    output = net.forward(x)

    loss = net.loss(output, target)
    losses << loss.value.value

    y_trues += target.argmax(axis: 1).to_a
    y_preds += output.value.argmax(axis: 1).to_a

    loss.backprop
    net.optimizer.update
  end

  accuracy = y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size

  puts "Epoch: #{epoch} | Accuracy: #{accuracy}"
end

Num::Plot::Plot.plot do
  scatter (0...losses.size), losses
  x_label "Epochs"
  y_label "Loss"
  label "MNIST Accuracy"
end
