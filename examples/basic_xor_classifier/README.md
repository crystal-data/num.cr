## Basic XOR Classifier

The following implements a simple XOR classifier to show how to use
`num.cr`'s `Network` class.  Plotting is done via `ishi`.

```crystal
require "num"
require "ishi"

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

m = x_train.shape[1]
epochs = 1500

options = {
  learning_rate: 0.1
}

costs = [] of Float64

net = Num::NN::Network(Float64).new(**options) do
  layer(3, 6, :tanh)
  layer(6, 1, :sigmoid)
end

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
```

The Network learns this function very quickly, as XOR is one of the simplest
distributions to hit.

```crystal
[[0.128285, 0.981884, 0.983019, 0.910526, 0.976077, 0.924345, 0.917729,
  0.312438]]
```

### Loss over time

![xorloss](static/xor_classifier_loss.png)
