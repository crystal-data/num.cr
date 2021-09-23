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

  def input(shape : Array(Int))
    @layers << Num::NN::InputLayer(T).new(@context, shape)
  end

  # Add a linear layer to the Network.  Since activation functions
  # are just treated as additional layers, this simply requires
  # the dimensions of the transformation.
  #
  # Dimensions should be `NUM_FEATURES` x `NUM_OUTPUTS`, so
  # if the data set is 100x10, with 200 neurons in the hidden layers,
  # the dimensions of the layer would be 10, 100, the 200 will be handled
  # by dynamically.
  #
  # Arguments
  # ---------
  # *i* : Int
  #   The number of features in the linear layer
  # *j* : Int
  #   The number of outputs in the linear layer
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  # end
  # ```
  def linear(output_size : Int)
    input_size = @layers.last.output_shape[0]
    @layers << Num::NN::LinearLayer(T).new(@context, input_size, output_size)
  end

  # Convolution layer for a neural network
  #
  # Arguments
  # ---------
  # shape : Array(Int32)
  #   Shape of the input to the layer
  # n : Int32
  #   Number of filters to apply
  # kh : Int32
  #   Filter height
  # kw : Int32
  #   Filter width
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def conv2d(n : Int32, kh : Int32, kw : Int32)
    input_shape = @layers.last.output_shape
    @layers << Num::NN::ConvolutionalLayer(T).new(@context, input_shape, n, kh, kw)
  end

  # Maxpool layer for a neural network
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Brief description of shape : Array(Int)
  # kernel : Tuple(Int
  #   Brief description of kernel : Tuple(Int
  # Int)
  #   Brief description of Int)
  # padding : Tuple(Int
  #   Brief description of padding : Tuple(Int
  # Int)
  #   Brief description of Int)
  # stride : Tuple(Int
  #   Brief description of stride : Tuple(Int
  # Int)
  #   Brief description of Int)
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def maxpool(kernel : Tuple(Int, Int), padding : Tuple(Int, Int), stride : Tuple(Int, Int))
    shape = @layers.last.output_shape
    @layers << Num::NN::MaxPoolLayer(T).new(@context, shape, kernel, padding, stride)
  end

  def dropout(prob : Float32 = 0.5_f32)
    shape = @layers.last.output_shape
    @layers << Num::NN::DropoutLayer(T).new(@context, shape, prob)
  end

  def softmax_cross_entropy_loss
    @loss = Num::NN::SoftmaxCrossEntropyLoss(T).new
  end

  # Adds a Flattening layer to a neural network
  #
  # Arguments
  # ---------
  # shape : Array(Int32)
  #   Shape of input to the layer
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def flatten
    shape = @layers.last.output_shape
    @layers << Num::NN::FlattenLayer(T).new(@context, shape)
  end

  # Add a ReLU layer to the Network.  Activation functions are handled
  # the same way as other layers, but do not change the dimensions
  # of the input
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  #   relu
  # end
  # ```
  def relu
    shape = @layers.last.output_shape
    @layers << Num::NN::ReluLayer(T).new(@context, shape)
  end

  # Adds a Leaky ReLU layer to a network.
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def leaky_relu
    shape = @layers.last.output_shape
    @layers << Num::NN::LeakyReluLayer(T).new(@context, shape)
  end

  # Adds an ELU layer to the network
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def elu
    shape = @layers.last.output_shape
    @layers << Num::NN::EluLayer(T).new(@context, shape)
  end

  # Add a Sigmoid layer to the Network.  Activation functions are handled
  # the same way as other layers, but do not change the dimensions
  # of the input
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  #   sigmoid
  # end
  # ```
  def sigmoid
    shape = @layers.last.output_shape
    @layers << Num::NN::SigmoidLayer(T).new(@context, shape)
  end

  # Add an SGD optimizer to the Network.
  #
  # Arguments
  # ---------
  # *learning_rate* : Float64
  #   Learning rate for all layers in the Network
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  #   sigmoid
  #   linear 3, 1
  #   sgd 0.7
  # end
  # ```
  def sgd(learning_rate : Float64 = 0.01)
    @optimizer = Num::NN::SGDOptimizer(T).new(learning_rate)
  end

  def adam(*args)
    @optimizer = Num::NN::AdamOptimizer(T).new(*args)
  end

  # Uses Sigmoid Cross Entropy to compute the loss for
  # the Network
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  #   sigmoid
  #   linear 3, 1
  #   sgd 0.7
  #   sigmoid_cross_entropy_loss
  # end
  # ```
  def sigmoid_cross_entropy_loss
    @loss = Num::NN::SigmoidCrossEntropyLoss(T).new
  end

  # Uses Mean Squared Error to compute the loss for
  # the Network
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # net = Num::NN::Network.new(ctx) do
  #   linear 2, 3
  #   sigmoid
  #   linear 3, 1
  #   sgd 0.7
  #   mse_loss
  # end
  # ```
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
  getter layers : Num::NN::NetworkInfo(T)

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

  # Uses the Network's loss function to calculate the loss
  # based on the final output from the Network, as well
  # as the target output
  #
  # Arguments
  # ---------
  # *output* : Num::Grad::Variable(T)
  #   Prediction by the network
  # *target* : T
  #   Tensor containing ground truth values
  #
  # Examples
  # --------
  # ```
  # epochs.times do |epoch|
  #   y_pred = net.forward(x)
  #   loss = net.loss(y_pred, y_actual)
  # end
  # ```
  def loss(output : Num::Grad::Variable(T), target : T)
    @layers.loss.loss(output, target)
  end

  # Return the Network's optimizer to allow updating
  # the weights and biases of the network
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # epochs.times do |epoch|
  #   y_pred = net.forward(x)
  #   loss = net.loss(y_pred, y_actual)
  #   net.optimizer.update
  # end
  # ```
  def optimizer
    @layers.optimizer
  end
end
