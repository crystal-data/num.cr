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

# :nodoc:
abstract class Num::NN::Layer(T)
  # :nodoc:
  def initialize(i : Int, j : Int, rate : Float = 0.001, dtype : T.class = T)
  end

  # :nodoc:
  abstract def forward(a_prev : Tensor(T)) : Tensor(T)

  # :nodoc:
  abstract def backward(d_a) : Tensor(T)
end

# :nodoc:
class Num::NN::ActivationLayer(T) < Num::NN::Layer(T)
  @w : Tensor(T)
  @b : Tensor(T)
  property rate : Float64
  @a_prev : Tensor(T)
  @z : Tensor(T)
  @a : Tensor(T)

  # :nodoc:
  def initialize(i : Int, j : Int, rate : Float = 0.1, dtype : T.class = T)
    @w = Tensor.random(T.new(0)...T.new(1), [j, i])
    @b = Tensor(T).zeros([j, 1])
    @rate = rate.to_f

    # These variables in this state are never used, so to
    # save memory they are just pointed to an existing
    # Tensor
    @a_prev = @w
    @z = @w
    @a = @w
  end

  # :nodoc:
  def forward(a_prev : Tensor(T)) : Tensor(T)
    @a_prev = a_prev
    @z = @w.matmul(@a_prev) + @b
    @a = self.activate(@z)
    @a
  end

  # :nodoc:
  def activate(input : Tensor(T)) : Tensor(T)
    input
  end

  # :nodoc:
  def backward(d_a) : Tensor(T)
    dz = self.derive(@z) * d_a
    dw = 1 / dz.shape[1] * dz.matmul(@a_prev.transpose)
    db = 1 / dz.shape[1] * dz.sum(axis: 1, dims: true)
    da_prev = @w.transpose.matmul(dz)
    @w.map!(dw) do |i, j|
      i - @rate * j
    end
    @b.map!(db) do |i, j|
      i - @rate * j
    end
    da_prev
  end

  # :nodoc:
  def derive(input : Tensor(T)) : Tensor(T)
    input
  end
end

class Num::NN::SigmoidLayer(T) < Num::NN::ActivationLayer(T)
  def activate(input : Tensor(T)) : Tensor(T)
    Num::NN.sigmoid(input)
  end

  def derive(input : Tensor(T)) : Tensor(T)
    Num::NN.d_sigmoid(input)
  end
end

class Num::NN::TanhLayer(T) < Num::NN::ActivationLayer(T)
  def activate(input : Tensor(T)) : Tensor(T)
    Num::NN.tanh(input)
  end

  def derive(input : Tensor(T)) : Tensor(T)
    Num::NN.d_tanh(input)
  end
end
