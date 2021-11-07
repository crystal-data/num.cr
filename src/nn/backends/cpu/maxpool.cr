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
  # Computes the maxpooling of a `Tensor`
  #
  # ## Arguments
  #
  # * input : `Tensor` - `Tensor` to pool
  # * kernel : Tuple - Kernel height and width
  # * target : `Tensor` - `Tensor` truth values
  # * padding : `Tuple` - `Tuple` with height and width of the padding
  # * stride : `Tuple` - `Tuple` with height and width of the stride
  def maxpool(
    input : Tensor(U, CPU(U)),
    kernel : Tuple(Int, Int),
    padding = {0, 0},
    stride = {0, 0}
  ) : Tuple(Tensor(Int32, CPU(Int32)), Tensor(U, CPU(U))) forall U
    nn = input.shape[0]
    cc = input.shape[1]
    hh = input.shape[2]
    ww = input.shape[3]

    kh = kernel[0]
    kw = kernel[1]

    outh = (hh + (2 * padding[0]) - kh) // stride[0] + 1
    outw = (ww + (2 * padding[1]) - kw) // stride[1] + 1

    max_indices = Tensor(Int32, CPU(Int32)).new([nn * cc * outh * outw])
    maxpooled = Tensor(U, CPU(U)).new([nn, cc, outh, outw])

    idata = input.to_unsafe
    idx_data = max_indices.to_unsafe
    max_data = maxpooled.to_unsafe

    nn.times do |n|
      cc.times do |c|
        outh.times do |h|
          outw.times do |w|
            max = U::MIN
            argmax = Int32::MIN

            kh.times do |ph|
              row = h * stride[0] + ph - padding[0]
              if (0 <= row) && (row < hh)
                kw.times do |pw|
                  col = w * stride[1] + pw - padding[1]

                  if (0 <= col) && (col < ww)
                    iidx = col + ww * (row + hh * (c + n * cc))
                    val = idata[iidx]
                    if val > max
                      max = val
                      argmax = iidx
                    end
                  end
                end
              end
            end

            oidx = w + outw * (h + outh * (c + n * cc))
            max_data[oidx] = max
            idx_data[oidx] = argmax
          end
        end
      end
    end
    {max_indices, maxpooled}
  end

  # Computes the maxpooling gradient
  #
  # ## Arguments
  #
  # * shape : `Array` - Shape of gradient output
  # * max_indices : `Tensor` - Pooled max indices
  # * grad_output : `Tensor` - Output from forward pass
  def maxpool_backward(
    shape : Array(Int),
    max_indices : Tensor(Int32, CPU(Int32)),
    grad_output : Tensor(U, CPU(U))
  ) : Tensor(U, CPU(U)) forall U
    unless grad_output.size == max_indices.size
      raise "Invalid shape"
    end

    result = Tensor(U, CPU(U)).zeros(shape)
    rdata = result.to_unsafe
    godata = grad_output.to_unsafe
    cmidata = max_indices.to_unsafe

    (grad_output.size).times do |i|
      rdata[cmidata[i]] = godata[i]
    end
    result
  end
end
