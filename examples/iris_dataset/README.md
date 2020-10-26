## Iris Dataset Classifier

The following implements a simple Iris dataset classifier to show how to use
`num.cr` on a multi-class problem with feature scaling.

```crystal
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

    y_trues += target.shape[0].times.map { |i| target[i].to_a.index(target[i].to_a.max) }.to_a
    y_preds += output.value.shape[0].times.map { |i| output.value[i].to_a.index(output[i].value.to_a.max) }.to_a

    loss.backprop
    net.optimizer.update
  end

  puts "Epoch: #{epoch} | Accuracy: #{y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size}"
end
```