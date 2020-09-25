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

module Num::NN
  def conv2d(
    input : Tensor(Float32),
    weight : Tensor(Float32),
    bias : Tensor(Float32),
    padding : Tuple(Int, Int),
    stride : Tuple(Int, Int) = {1, 1}
  )
    batch_size = input.shape[0]
    output_channels = weight.shape[-4]
    output_height = (2 * padding[0] + input.shape[-2]) - (weight.shape[-2] - 1)
    output_width = (2 * padding[1] + input.shape[-1]) - (weight.shape[-1] - 1)

    input = input.dup(Num::RowMajor)
    weight = weight.dup(Num::RowMajor)

    result = Tensor(Float32).new([input.shape[0], output_channels, output_height, output_width])

    status = LibNNPACK.nnp_convolution_output(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size.to_u64,
      input.shape[-3].to_u64,
      output_channels.to_u64,
      LibNNPACK::NNPSize.new(height: input.shape[-2], width: input.shape[-1]),
      LibNNPACK::NNPPadding.new(top: padding[0], bottom: padding[0], left: padding[1], right: padding[1]),
      LibNNPACK::NNPSize.new(height: weight.shape[-2], width: weight.shape[-1]),
      input.to_unsafe,
      weight.to_unsafe,
      bias.to_unsafe,
      result.to_unsafe,
      nil,
      nil,
      LibNNPACK::NNPActivation::IDENTITY,
      nil,
      nil,
      out profile,
    )

    unless status == LibNNPACK::NNPStatus::SUCCESS
      raise Exception.new "NNPACK failed with #{status}.  Did you run with the -Dnnpack flag?"
    end

    result
  end

  def conv2d_backward(
    input : Tensor(Float32),
    weight : Tensor(Float32),
    bias : Tensor(Float32),
    grad_output : Tensor(Float32),
    padding : Tuple(Int, Int),
    stride : Tuple(Int, Int) = {1, 1}
  )
    batch_size = input.shape[0]
    input_channels = input.shape[-3]
    output_channels = weight.shape[-4]
    output_height = (2 * padding[0] + input.shape[-2]) - (weight.shape[-2] - 1)
    output_width = (2 * padding[1] + input.shape[-1]) - (weight.shape[-1] - 1)

    nninput_size = LibNNPACK::NNPSize.new(height: input.shape[-2], width: input.shape[-1])
    nnpadding = LibNNPACK::NNPPadding.new(top: padding[0], bottom: padding[0], left: padding[1], right: padding[1])
    nnkernel_size = LibNNPACK::NNPSize.new(height: weight.shape[-2], width: weight.shape[-1])

    grad_input = Tensor(Float32).zeros(input.shape)

    status = LibNNPACK.nnp_convolution_input_gradient(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size,
      input_channels,
      output_channels,
      nninput_size,
      nnpadding,
      nnkernel_size,
      grad_output.to_unsafe,
      weight.to_unsafe,
      input.to_unsafe,
      nil,
      nil,
      LibNNPACK::NNPActivation::IDENTITY,
      nil,
      nil,
      out input_profile
    )

    unless status == LibNNPACK::NNPStatus::SUCCESS
      raise Exception.new "NNPACK failed with #{status}.  Did you run with the -Dnnpack flag?"
    end

    grad_weight = Tensor(Float32).zeros(input.shape)

    status = LibNNPACK.nnp_convolution_kernel_gradient(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size,
      input_channels,
      output_channels,
      nninput_size,
      nnpadding,
      nnkernel_size,
      input.to_unsafe,
      grad_output.to_unsafe,
      grad_weight.to_unsafe,
      nil,
      nil,
      LibNNPACK::NNPActivation::IDENTITY,
      nil,
      nil,
      out weight_profile
    )

    unless status == LibNNPACK::NNPStatus::SUCCESS
      raise Exception.new "NNPACK failed with #{status}.  Did you run with the -Dnnpack flag?"
    end

    grad_bias = bias
    if bias.rank == 3
      grad_bias = grad_bias.sum(3).sum(2).sum(0).reshape(bias.shape)
    end

    {grad_input, grad_weight, grad_bias}
  end
end
