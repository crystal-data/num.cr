# Copyright (c) 2021 Crystal Data Contributors
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
  extend self

  # Computes a 2D convolution over input images. Intended to be used
  # in 2d convolution forward pass. This applies a 2D cross-correlation,
  # not to be confused with the mathematical convolution.
  #
  # ## Arguments
  #
  # * input : `Tensor` - 4D `Tensor` batch of images of the size [N,C_in,H_in,W_in]
  # * weight : `Tensor` - 4D `Tensor` convolving kernel weights of the size [C_out,C_in,kH,kW]
  # * bias : `Tensor` - 3D `Tensor` bias of the size [C_out,1,1]
  # * padding : `Tuple` - `Tuple` with height and width of the padding
  # * stride : `Tuple` - `Tuple` with height and width of the stride
  def conv2d(
    input : Tensor(Float32, CPU(Float32)),
    weight : Tensor(Float32, CPU(Float32)),
    bias : Tensor(Float32, CPU(Float32)),
    padding : Tuple(Int, Int),
    stride : Tuple(Int, Int) = {1, 1}
  )
    batch_size = input.shape[0]
    output_channels = weight.shape[-4]
    output_height = (2 * padding[0] + input.shape[-2]) - (weight.shape[-2] - 1)
    output_width = (2 * padding[1] + input.shape[-1]) - (weight.shape[-1] - 1)

    result = Tensor(Float32, CPU(Float32)).new([input.shape[0], output_channels, output_height, output_width])

    status = LibNNPACK.nnp_convolution_output(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size.to_u64,
      input.shape[-3].to_u64,
      output_channels.to_u64,
      LibNNPACK::NNPSize.new(height: input.shape[-2], width: input.shape[-1]),
      LibNNPACK::NNPPadding.new(top: padding[0], bottom: padding[0], left: padding[1], right: padding[1]),
      LibNNPACK::NNPSize.new(height: weight.shape[-2], width: weight.shape[-1]),
      input.get_offset_ptr,
      weight.get_offset_ptr,
      bias.get_offset_ptr,
      result.get_offset_ptr,
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

  # Computes gradients of a 2D convolution. Intended to be used after
  # `conv2d` to calculate gradients in backward pass.
  #
  # ## Arguments
  #
  # * input : `Tensor` - 4D `Tensor` batch of images of the size [N,C_in,H_in,W_in]
  # * weight : `Tensor` - 4D `Tensor` convolving kernel weights of the size [C_out,C_in,kH,kW]
  # * bias : `Tensor` - 3D `Tensor` bias of the size [C_out,1,1]
  # * grad_output : `Tensor` - 4D `Tensor` gradient of size [N, C_out, H_out, W_out]
  # * padding : `Tuple` - `Tuple` with height and width of the padding
  # * stride : `Tuple` - `Tuple` with height and width of the stride
  def conv2d_backward(
    input : Tensor(Float32, CPU(Float32)),
    weight : Tensor(Float32, CPU(Float32)),
    bias : Tensor(Float32, CPU(Float32)),
    grad_output : Tensor(Float32, CPU(Float32)),
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

    grad_input = Tensor(Float32, CPU(Float32)).zeros(input.shape)

    status = LibNNPACK.nnp_convolution_input_gradient(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size,
      input_channels,
      output_channels,
      nninput_size,
      nnpadding,
      nnkernel_size,
      grad_output.get_offset_ptr,
      weight.get_offset_ptr,
      grad_input.get_offset_ptr,
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

    grad_weight = Tensor(Float32, CPU(Float32)).zeros(weight.shape)

    status = LibNNPACK.nnp_convolution_kernel_gradient(
      LibNNPACK::NNPConvolutionAlgorithm::AUTO,
      batch_size,
      input_channels,
      output_channels,
      nninput_size,
      nnpadding,
      nnkernel_size,
      input.get_offset_ptr,
      grad_output.get_offset_ptr,
      grad_weight.get_offset_ptr,
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

    grad_bias = grad_output.sum(3).sum(2).sum(0).reshape(bias.shape)

    {grad_input, grad_weight, grad_bias}
  end

  # :nodoc:
  def im2col(
    input : Tensor(U, CPU(U)),
    kernel : Tuple(Int, Int),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U, CPU(U)) forall U
    channels = input.shape[-3]
    height = input.shape[-2]
    width = input.shape[-1]

    channels_col = channels * kernel[0] * kernel[1]
    height_col = (height + (2 * padding[0]) - kernel[0]) // stride[0] + 1
    width_col = (width + (2 * padding[1]) - kernel[1]) // stride[1] + 1
    flatten_size_col = height_col * width_col
    flatten_size = height * width

    result = Tensor(U, CPU(U)).new([channels_col, flatten_size_col])

    odata = result.get_offset_ptr
    idata = input.get_offset_ptr

    channels_col.times do |c|
      w_offset = (c % kernel[1]) - padding[1]
      h_offset = ((c // kernel[1]) % kernel[0]) - padding[0]
      c_offset = (c // kernel[1]) // kernel[0]

      height_col.times do |h|
        row = h_offset + (h * stride[0])
        offset_col = h * width_col

        width_col.times do |w|
          col = w_offset + (w * stride[1])
          v = U.new(0)

          if row >= 0 && col >= 0 && row < height && col < width
            iidx = (c_offset * flatten_size) + row * width + col
            v = idata[iidx]
          end

          oidx = (c * flatten_size_col) + offset_col + w
          odata[oidx] = v
        end
      end
    end

    result
  end

  # :nodoc:
  def col2im(
    input : Tensor(U, CPU(U)),
    channels : Int,
    height : Int,
    width : Int,
    kernel_size : Tuple(Int, Int),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U, CPU(U)) forall U
    channels_col = channels * kernel_size[0] * kernel_size[1]
    height_col = (height + (2 * padding[0]) - kernel_size[0]) // stride[0] + 1
    width_col = (width + (2 * padding[1]) - kernel_size[1]) // stride[1] + 1
    result = Tensor(U, CPU(U)).zeros([channels, height, width])

    channels_col.times do |c|
      w_offset = (c % kernel_size[1]) - padding[1]
      h_offset = ((c // kernel_size[1]) % kernel_size[0]) - padding[0]
      c_offset = (c // kernel_size[1]) // kernel_size[0]
      height_col.times do |h|
        row = h_offset + (h * stride[0])
        offset_col = h * width_col
        width_col.times do |w|
          col = w_offset + (w * stride[1])
          if row < 0 || col < 0 || row >= height || col >= width
            next
          end
          result[c_offset, row, col].map!(input[c, offset_col + w]) do |i, j|
            i + j
          end
        end
      end
    end

    result
  end

  # Computes a 2D convolution over input images. Intended to be used
  # in 2d convolution forward pass. This applies a 2D cross-correlation,
  # not to be confused with the mathematical convolution.
  #
  # ## Arguments
  #
  # * input : `Tensor` - 4D `Tensor` batch of images of the size [N,C_in,H_in,W_in]
  # * weight : `Tensor` - 4D `Tensor` convolving kernel weights of the size [C_out,C_in,kH,kW]
  # * bias : `Tensor` - 3D `Tensor` bias of the size [C_out,1,1]
  # * padding : `Tuple` - `Tuple` with height and width of the padding
  # * stride : `Tuple` - `Tuple` with height and width of the stride
  def im2colgemm_conv2d(
    input : Tensor(U, CPU(U)),
    kernel : Tensor(U, CPU(U)),
    bias : Tensor(U, CPU(U)),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U, CPU(U)) forall U
    batch_size = input.shape[-4]
    output_channels = kernel.shape[-4]
    kernel_size = {kernel.shape[-2], kernel.shape[-1]}
    output_height = (input.shape[-2] + (2 * padding[0]) - kernel.shape[-2]) // stride[0] + 1
    output_width = (input.shape[-1] + (2 * padding[1]) - kernel.shape[-1]) // stride[1] + 1
    channels_col = input.shape[-3] * kernel.shape[-2] * kernel.shape[-1]
    kernel_col = kernel.reshape(output_channels, channels_col)

    result = Tensor(U, CPU(U)).new([batch_size, output_channels, output_height, output_width])

    batch_size.times do |i|
      input_col = im2col(input[i], kernel_size, padding, stride)
      output = result[i].reshape(kernel_col.shape[0], input_col.shape[1])
      kernel_col.matmul(input_col, output)
    end
    result + bias
  end

  # Computes gradients of a 2D convolution. Intended to be used after
  # `conv2d` to calculate gradients in backward pass.
  #
  # ## Arguments
  #
  # * input : `Tensor` - 4D `Tensor` batch of images of the size [N,C_in,H_in,W_in]
  # * weight : `Tensor` - 4D `Tensor` convolving kernel weights of the size [C_out,C_in,kH,kW]
  # * bias : `Tensor` - 3D `Tensor` bias of the size [C_out,1,1]
  # * grad_output : `Tensor` - 4D `Tensor` gradient of size [N, C_out, H_out, W_out]
  # * padding : `Tuple` - `Tuple` with height and width of the padding
  # * stride : `Tuple` - `Tuple` with height and width of the stride
  def im2colgemm_conv2d_gradient(
    input : Tensor(U, CPU(U)),
    kernel : Tensor(U, CPU(U)),
    bias : Tensor(U, CPU(U)),
    grad_output : Tensor(U, CPU(U)),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tuple(Tensor(U, CPU(U)), Tensor(U, CPU(U)), Tensor(U, CPU(U))) forall U
    batch_size = input.shape[-4]
    output_channels = kernel.shape[-4]
    kernel_size = {kernel.shape[-2], kernel.shape[-1]}
    output_height = (input.shape[-2] + (2*padding[0]) - kernel.shape[-2]) // stride[0] + 1
    output_width = (input.shape[-1] + (2*padding[1]) - kernel.shape[-1]) // stride[1] + 1
    output_flatten_size = output_height * output_width
    channels_col = input.shape[-3] * kernel_size[0] * kernel_size[1]
    kernel_col = kernel.reshape(output_channels, input.shape[-3] * kernel.shape[-2] * kernel.shape[-1])

    grad_input = Tensor(U, CPU(U)).zeros([batch_size, input.shape[-3], input.shape[-2], input.shape[-1]])
    grad_weight = Tensor(U, CPU(U)).zeros([output_channels, kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]])
    grad_bias = grad_output.sum(3).sum(2).sum(0).reshape(bias.shape)

    batch_size.times do |i|
      grad_output_col = grad_output[i].reshape(output_channels, output_flatten_size)
      grad_input_col = kernel_col.transpose.matmul(grad_output_col)

      input_col = im2col(input[i], kernel_size, padding, stride)
      grad_input[i] = col2im(grad_input_col, input.shape[-3], input.shape[-2], input.shape[-1], kernel_size, padding, stride)
      grad_weight.map!((grad_output_col.matmul(input_col.transpose)).reshape(grad_weight.shape)) do |i, j|
        i + j
      end
    end

    {grad_input, grad_weight, grad_bias}
  end
end
