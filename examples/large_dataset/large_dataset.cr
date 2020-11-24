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

# Replace this with your rock/paper/scissors dataset location
IMAGE_PATH = "/Users/chris/Downloads/archive"

class RPSStream < Num::NN::DataStream
  def self.build(path : String, width : Int32, height : Int32, num_classes : Int32, shuffle : Bool = true, batch_size : Int32 = 32) : Num::NN::DataStream
    base_path = Path.new path
    category_mapping = {"rock" => 0, "scissors" => 1, "paper" => 2}
    mapping = Hash(String, Int32).new

    category_mapping.each_key do |k|
      d = Dir.new base_path / k
      d.each_child do |f|
        filename = base_path / k / f
        mapping[filename.to_s] = category_mapping[k]
      end
    end

    new(mapping, width, height, num_classes, shuffle, batch_size)
  end
end

stream = RPSStream.build IMAGE_PATH, 300, 200, 3
ctx = Num::Grad::Context(Tensor(Float32)).new

net = Num::NN::Network.new(ctx) do
  input [1, 300, 200]
  conv2d 20, 5, 5
  relu
  maxpool({2, 2}, {0, 0}, {2, 2})
  conv2d 20, 5, 5
  maxpool({2, 2}, {0, 0}, {2, 2})
  flatten
  linear 10
  relu
  linear 3
  softmax_cross_entropy_loss
  sgd 0.01
end

100.times do |epoch|
  stream.each do |batch, truth|
    x_train = ctx.variable(batch)
    x_pred = net.forward(x_train)
    puts x_pred[0]
    puts truth[0]
    loss = net.loss(x_pred, truth)
    loss.backprop
    net.optimizer.update
  end
end
