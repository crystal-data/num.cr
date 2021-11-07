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
  # Computes a forward dropout activation
  #
  # ## Arguments
  #
  # * input : `Tensor` - `Tensor` to activate
  # * mask : `Tensor` - Mask to dropout
  # * probability : `Float` - Probability of dropout
  def dropout(
    input : Tensor(U, OCL(U)),
    mask : Tensor(U, OCL(U)),
    probability : Float
  ) : Tensor(U, OCL(U)) forall U
    input * mask / U.new(probability)
  end

  # Computes a backwards dropout derivative
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` used to compute backwards pass
  # * mask : `Tensor` - Mask to apply to the gradient
  # * probability : `Float` - Probability of dropout
  def dropout_backwards(
    gradient : Tensor(U, CPU(U)),
    mask : Tensor(U, CPU(U)),
    probability : Float
  ) : Tensor(U, OCL(U)) forall U
    gradient * mask / U.new(probability)
  end
end
