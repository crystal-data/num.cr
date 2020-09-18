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

# A simple wrapper around an array that provides a
# `layer` method for easy network construction
# within a block of data
class Num::NN::LayerArray(T)
  getter layers : Array(Num::NN::Layer(T))

  @@layer_mapping = {
    tanh:    Num::NN::TanhLayer,
    sigmoid: Num::NN::SigmoidLayer,
  }

  # This should always be initialized with an empty
  # array of layers that can be tapped and yielded
  # by Network creation
  def initialize
    @layers = [] of Num::NN::Layer(T)
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
  def layer(i : Int, j : Int, act : Symbol)
    @layers << @@layer_mapping[act].new(i, j, dtype: T)
  end

  def layer(i : Int, j : Int, act : U.class) forall U
    act.new(i, j, dtype: T)
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
  @layers : LayerArray(T)

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
  def self.new(**options)
    layers = LayerArray(T).new
    layers.tap do |instance|
      with instance yield
    end
    new(layers, **options)
  end

  # :nodoc:
  private def initialize(@layers : LayerArray(T), **options)
    if options.has_key?(:learning_rate)
      @layers.each do |layer|
        layer.rate = options[:learning_rate]
      end
    end
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
  def forward(train : Tensor(T)) : Tensor(T)
    @layers.each do |layer|
      train = layer.forward(train)
    end
    train
  end

  # Propogates an error back through a network, returning the first
  # gradient from the network
  #
  # Arguments
  # ---------
  # *train* : Tensor(T)
  #   Error gradient
  #
  # Examples
  # --------
  # ```
  # a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].to_tensor
  # a = net.forward(a)
  #
  # g = computeLoss(a)
  # net.backward(g)
  # ```
  def backward(gradient) : Tensor(T)
    @layers.reverse_each do |layer|
      gradient = layer.backward(gradient)
    end
    gradient
  end
end
