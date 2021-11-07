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

  # Represents the type of Fan meant to be discovered,
  # whether it is the maximum number of inputs or
  # outputs.
  enum FanMode
    FanAvg
    FanIn
    FanOut
  end

  # Type of distribution to return from the random
  # initial values
  enum Distribution
    Uniform
    Normal
  end

  def compute_fans(*shape : Int)
    f0 = shape[0]
    f1 = shape[1]
    return {f0, f1} unless shape.size != 1

    product = shape[1...].reduce(1) { |i, j| i * j }
    f0 *= product
    f1 *= product

    {f0, f1}
  end

  def variance_scaled(
    *shape : Int,
    dtype : U.class,
    device : V.class,
    scale : U = U.new(1),
    mode : FanMode = FanMode::FanIn,
    distribution : Distribution = Distribution::Normal
  ) forall U, V
    f0, f1 = compute_fans(*shape)

    std = case mode
          when FanMode::FanIn
            Math.sqrt(scale / U.new(f0))
          when FanMode::FanOut
            Math.sqrt(scale / U.new(f1))
          else
            Math.sqrt(scale * U.new(2) / U.new(f0 + f1))
          end

    case distribution
    when Distribution::Uniform
      limit = Math.sqrt(U.new(3)) * std
      Tensor.random(-limit...limit, shape.to_a, device: V)
    else
      Tensor(U, V).normal(shape.to_a, U.new(0), std)
    end
  end

  def kaiming_uniform(*shape : Int, dtype : Tensor(U, V).class) forall U, V
    variance_scaled(
      *shape,
      dtype: U,
      device: V,
      scale: U.new(2),
      mode: FanMode::FanIn,
      distribution: Distribution::Uniform
    )
  end

  def kaiming_normal(*shape : Int, dtype : Tensor(U, V).class) forall U, V
    variance_scaled(
      *shape,
      dtype: U,
      device: V,
      scale: U.new(2),
      mode: FanMode::FanIn,
      distribution: Distribution::Normal
    )
  end
end
