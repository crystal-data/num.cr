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

  enum FanMode
    FanAvg
    FanIn
    FanOut
  end

  enum Distribution
    Uniform
    Normal
  end

  def compute_fans(*shape : Int)
    fan_out = shape[0]
    fan_in = shape[1]

    if shape.size == 1
      return {fan_out, fan_in}
    end

    product = 1
    2.step(to: shape.size - 1) do |i|
      product *= shape[i]
    end

    fan_out *= product
    fan_in *= product

    {fan_out, fan_in}
  end

  def variance_scaled(*shape : Int, dtype : U.class, scale : U = U.new(1), mode : FanMode = FanMode::FanIn, distribution : Distribution = Distribution::Normal) forall U
    fan_in, fan_out = compute_fans(*shape)
    case mode
    when FanMode::FanIn
      std = Math.sqrt(scale / U.new(fan_in))
    when FanMode::FanOut
      std = Math.sqrt(scale / U.new(fan_out))
    else
      std = Math.sqrt(scale * U.new(2) / U.new(fan_in + fan_out))
    end

    case distribution
    when Distribution::Uniform
      limit = Math.sqrt(U.new(3)) * std
      Tensor.random(-limit...limit, shape.to_a)
    else
      Tensor(U).normal(shape.to_a, U.new(0), std)
    end
  end

  def kaiming_uniform(*shape : Int, dtype : Tensor(U).class) forall U
    variance_scaled(*shape, dtype: U, scale: U.new(2), mode: FanMode::FanIn, distribution: Distribution::Uniform)
  end

  def kaiming_normal(*shape : Int, dtype : Tensor(U).class) forall U
    variance_scaled(*shape, dtype: U, scale: U.new(2), mode: FanMode::FanIn, distribution: Distribution::Normal)
  end
end
