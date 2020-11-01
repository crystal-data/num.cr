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
  extend self

  def im2col(
    input : Tensor(U),
    kernel : Tuple(Int, Int),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U) forall U
    channels = input.shape[-3]
    height = input.shape[-2]
    width = input.shape[-1]

    channels_col = channels * kernel[0] * kernel[1]
    height_col = (height + (2 * padding[0]) - kernel[0]) // stride[0] + 1
    width_col = (width + (2 * padding[1]) - kernel[1]) // stride[1] + 1
    flatten_size_col = height_col * width_col
    flatten_size = height * width

    result = Tensor(U).new([channels_col, flatten_size_col])

    odata = result.to_unsafe
    idata = input.to_unsafe

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

  def col2im(
    input : Tensor(U),
    channels : Int,
    height : Int,
    width : Int,
    kernel_size : Tuple(Int, Int),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U) forall U
    channels_col = channels * kernel_size[0] * kernel_size[1]
    height_col = (height + (2 * padding[0]) - kernel_size[0]) // stride[0] + 1
    width_col = (width + (2 * padding[1]) - kernel_size[1]) // stride[1] + 1
    result = Tensor(U).zeros([channels, height, width])

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

  def im2colgemm_conv2d(
    input : Tensor(U),
    kernel : Tensor(U),
    bias : Tensor(U),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tensor(U) forall U
    batch_size = input.shape[-4]
    output_channels = kernel.shape[-4]
    kernel_size = {kernel.shape[-2], kernel.shape[-1]}
    output_height = (input.shape[-2] + (2 * padding[0]) - kernel.shape[-2]) // stride[0] + 1
    output_width = (input.shape[-1] + (2 * padding[1]) - kernel.shape[-1]) // stride[1] + 1
    channels_col = input.shape[-3] * kernel.shape[-2] * kernel.shape[-1]
    kernel_col = kernel.reshape(output_channels, channels_col)

    result = Tensor(U).new([batch_size, output_channels, output_height, output_width])

    batch_size.times do |i|
      input_col = im2col(input[i], kernel_size, padding, stride)
      output = result[i].reshape(kernel_col.shape[0], input_col.shape[1])
      kernel_col.matmul(input_col, output)
    end

    result
  end

  def im2colgemm_conv2d_gradient(
    input : Tensor(U),
    kernel : Tensor(U),
    bias : Tensor(U),
    grad_output : Tensor(U),
    padding : Tuple(Int, Int) = {0, 0},
    stride : Tuple(Int, Int) = {1, 1}
  ) : Tuple(Tensor(U), Tensor(U), Tensor(U)) forall U
    batch_size = input.shape[-4]
    output_channels = kernel.shape[-4]
    kernel_size = {kernel.shape[-2], kernel.shape[-1]}
    output_height = (input.shape[-2] + (2*padding[0]) - kernel.shape[-2]) // stride[0] + 1
    output_width = (input.shape[-1] + (2*padding[1]) - kernel.shape[-1]) // stride[1] + 1
    output_flatten_size = output_height * output_width
    channels_col = input.shape[-3] * kernel_size[0] * kernel_size[1]
    kernel_col = kernel.reshape(output_channels, input.shape[-3] * kernel.shape[-2] * kernel.shape[-1])

    grad_input = Tensor(U).zeros([batch_size, input.shape[-3], input.shape[-2], input.shape[-1]])
    grad_weight = Tensor(U).zeros([output_channels, kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]])

    batch_size.times do |i|
      grad_output_col = grad_output[i].reshape(output_channels, output_flatten_size)
      grad_input_col = kernel_col.transpose.matmul(grad_output_col)

      input_col = im2col(input[i], kernel_size, padding, stride)
      grad_input[i] = col2im(grad_input_col, input.shape[-3], input.shape[-2], input.shape[-1], kernel_size, padding, stride)
      grad_weight.map!((grad_output_col.matmul(input_col.transpose)).reshape(grad_weight.shape)) do |i, j|
        i + j
      end
    end

    grad_bias = grad_output.sum(3).sum(2).sum(0).reshape(bias.shape)
    {grad_input, grad_weight, grad_bias}
  end
end
