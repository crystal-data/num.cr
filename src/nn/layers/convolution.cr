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

# In a CNN, the input is a `Tensor` with a shape:
#
# (# of inputs) x (input h) x (input w) x (input channels)
#
# After passing through a convolutional layer, the image becomes abstracted
# to a feature map, also called an activation map, with shape:
#
# (# of inputs) x (feature map h) x (feature map w) x (feature map channels)
#
# Convolutional layers convolve the input and pass its result to the next
# layer. This is similar to the response of a neuron in the visual cortex
# to a specific stimulus. Each convolutional neuron processes data
# only for its receptive field. Although fully connected feedforward neural
# networks can be used to learn features and classify data, this architecture
# is generally impractical for larger inputs such as high resolution images.
# It would require a very high number of neurons, even in a shallow
# architecture, due to the large input size of images, where each pixel
# is a relevant input feature. For instance, a fully connected layer for a
# (small) image of size 100 x 100 has 10,000 weights for each neuron in the
# second layer. Instead, convolution reduces the number of free parameters,
# allowing the network to be deeper. For example, regardless of image size,
# using a 5 x 5 tiling region, each with the same shared weights, requires
# only 25 learnable parameters. Using regularized weights over fewer
# parameters avoids the vanishing gradients and exploding gradients problems
# seen during backpropagation in traditional neural networks. Furthermore,
# convolutional neural networks are ideal for data with a grid-like topology
# (such as images) as spatial relations between separate features are taken
# into account during convolution and/or pooling.
class Num::NN::ConvolutionalLayer(T) < Num::NN::Layer(T)
  @in_shape : Array(Int32)
  @weights : Num::Grad::Variable(T)
  @bias : Num::Grad::Variable(T)
  @padding : Tuple(Int32, Int32)
  @stride : Tuple(Int32, Int32) = {1, 1}

  # Creates a convolutional layer in a `Network`
  #
  # ## Arguments
  #
  # * context : `Num::Grad::Context(T)` - Context of the network.  This argument
  #   is used entirely to determine the generic type of the layer
  # * in_shape : `Array(Int)` - Shape of input to layer
  # * num_filters : `Int` - Number of filters to apply in the convolution
  # * kernel_height : `Int` - Height of kernel for convolution
  # * kernel_width : `Int` - Width of kernel for convolution
  # * padding : `Int` - Padding of kernel
  # * stride : `Int` - Stride of kernel.
  #
  # NOTE: The `stride` argument is currently only supported for the
  # `im2colgemm_conv2d`, as it is not supported by NNPACK.  Using this
  # parameter is rarely worth the large performance difference if you
  # are able to use NNPACK
  def initialize(
    context : Num::Grad::Context(T),
    in_shape : Array(Int),
    num_filters : Int,
    kernel_height : Int,
    kernel_width : Int,
    @padding = {0, 0},
    @stride = {1, 1}
  )
    @in_shape = in_shape.map &.to_i
    c_in, h_in, w_in = in_shape
    w = Num::NN.kaiming_normal(num_filters, c_in, kernel_height, kernel_width, dtype: T)
    b = T.zeros([num_filters, 1, 1])
    @weights = context.variable(w)
    @bias = context.variable(b)
  end

  # Performs a forward pass of a variable through a `ConvolutionalLayer`
  #
  # ## Arguments
  #
  # * input : `Num::Grad::Variable(T)` - Variable to convolve
  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    output = \
       {% if flag?(:nnpack) %}
         Num::NN.conv2d(input.value, @weights.value, @bias.value, @padding, @stride)
       {% elsif flag?(:im2col) %}
         Num::NN.im2colgemm_conv2d(input.value, @weights.value, @bias.value, @padding, @stride)
       {% else %}
         Num::NN.im2colgemm_conv2d(input.value, @weights.value, @bias.value, @padding, @stride)
       {% end %}

    result = input.context.variable(output)

    if input.is_grad_needed || @weights.is_grad_needed || @bias.is_grad_needed
      gate = Num::NN::ConvolutionGate.new(input, @weights, @bias, @padding, @stride)
      gate.cache(result, input, @weights, @bias, @padding, @stride)
    end
    result
  end

  # Returns all `Num::Grad::Variables` associated with the `Layer`.
  # Used primarily to register variables with optimizers
  def variables : Array(Num::Grad::Variable(T))
    [@weights, @bias]
  end

  # Returns the output shape of a `ConvolutionalLayer`.  This method is
  # primarily used to infer the input shape of following layers in
  # a `Network`
  def output_shape : Array(Int32)
    kh = @weights.value.shape[2]
    kw = @weights.value.shape[3]
    ph = @padding[0]
    pw = @padding[1]
    sh = @stride[0]
    sw = @stride[1]

    ih = @in_shape[1]
    iw = @in_shape[2]
    dh = 1
    dw = 1

    r0 = @weights.value.shape[0]
    r1 = 1 + (ih + 2 * ph - (((kh - 1) * dh) + 1)) // sh
    r2 = 1 + (iw + 2 * pw - (((kw - 1) * dw) + 1)) // sw
    [r0, r1, r2]
  end
end
