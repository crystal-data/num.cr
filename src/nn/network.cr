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

class Num::NN::NetworkInfo(T)
  getter layers : Array(Num::NN::Layer(T))
  getter optimizer : Num::NN::Optimizer(T)
  getter context : Num::Grad::Context(T)
  getter loss : Num::NN::Loss(T)

  # This should always be initialized with an empty
  # array of layers that can be tapped and yielded
  # by Network creation
  def initialize(@context : Num::Grad::Context(T))
    @layers = [] of Num::NN::Layer(T)
    @optimizer = Num::NN::Optimizer(T).new
    @loss = Num::NN::Loss(T).new
  end

  # Helper method to easily add layers to a Network.
  #
  # Rather than having to explicitly create layers and
  # add an array of them to a Network, this method is
  # provided by yielding its own instance to allow:
  #
  # ```
  # Network(Float32).new do
  #   layer(3, 2, :relu)
  # end
  # ```
  #
  # This should eventually be validated to make sure that layers
  # line up with each other in terms of input/output sizes
  def layer(cls : U.class, *args) forall U
    @layers << U.new(*args)
  end

  def linear(i : Int, j : Int)
    @layers << Num::NN::LinearLayer(T).new(@context, i, j)
  end

  def relu
    @layers << Num::NN::ReluLayer(T).new(@context)
  end

  def sigmoid
    @layers << Num::NN::SigmoidLayer(T).new(@context)
  end

  def sgd(learning_rate : Float64 = 0.01)
    @optimizer = Num::NN::SGDOptimizer(T).new(learning_rate)
  end

  def sigmoid_cross_entropy_loss
    @loss = Num::NN::SigmoidCrossEntropyLoss(T).new
  end

  def mse_loss
    @loss = Num::NN::MSELoss(T).new
  end

  forward_missing_to layers
end

# A neural network can be defined as a biologically inspired
# computational model that consists of a network architecture
# composed of artificial neurons. This structure contains a set of
# parameters, which can be adjusted to perform specific tasks.
#
# This class is a loose wrapper that primarily provides syntactic
# sugar around moving data through a network -> forward, and propogating
# loss <- backwards
class Num::NN::Network(T)
  @layers : Num::NN::NetworkInfo(T)

  # Convenience method to allow for creation of a Network
  # with as little code as possible.  Taps an instance of
  # a LayerArray in order to allow layers to be added to the
  # network in a block
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # Network(Float32).new do
  #   layer(2, 3, :tanh)
  #   layer(3, 1, :sigmoid)
  # end
  # ```
  def self.new(context : Num::Grad::Context(T), **options)
    layers = Num::NN::NetworkInfo(T).new(context)
    layers.tap do |instance|
      with instance yield
    end
    layers.optimizer.build_params(layers.layers)
    new(layers, **options)
  end

  # :nodoc:
  private def initialize(@layers : Num::NN::NetworkInfo(T), **options)
  end

  # Propogates an input through a network, returning
  # the final prediction from the network
  #
  # Arguments
  # ---------
  # *train* : Tensor(T)
  #   Training input data
  #
  # Examples
  # --------
  # ```
  # a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].to_tensor
  # net.forward(a)
  # ```
  def forward(train : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    @layers.each do |layer|
      train = layer.forward(train)
    end
    train
  end

  def loss(output : Num::Grad::Variable(T), target : T)
    @layers.loss.loss(output, target)
  end

  def optimizer
    @layers.optimizer
  end
end
