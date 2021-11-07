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

# Adam (short for Adaptive Moment Estimation) is an update to the
# RMSProp optimizer. In this optimization algorithm, running averages of
# both the gradients and the second moments of the gradients are used.
class Num::NN::AdamOptimizer(T) < Num::NN::Optimizer(T)
  @params : Array(Num::Grad::Variable(T))
  @learning_rate : Float64
  @beta1 : Float64
  @beta2 : Float64
  @beta1_t : Float64
  @beta2_t : Float64
  @first_moments : Array(T)
  @second_moments : Array(T)
  @epsilon : Float64

  # Initializes an Adam optimizer, disconnected from a network.
  # In order to link this optimizer to a `Num::NN::Network`, calling
  # `build_params` will register each variable in the computational
  # graph with this optimizer.
  #
  # ## Arguments
  #
  # * learning_rate : `Float` - Learning rate of the optimizer
  # * beta1 : `Float` - The exponential decay rate for the 1st moment estimates
  # * beta2 : `Float` - The exponential decay rate for the 2nd moment estimates
  # * epsilon : `Float` - A small constant for numerical stability
  def initialize(
    @learning_rate : Float64 = 0.001,
    @beta1 : Float64 = 0.9,
    @beta2 : Float64 = 0.999,
    @epsilon : Float64 = 1e-8
  )
    @params = [] of Num::Grad::Variable(T)
    @beta1_t = @beta1
    @beta2_t = @beta2

    @first_moments = [] of T
    @second_moments = [] of T
  end

  # Adds variables from a `Num::NN::Network` to the optimizer,
  # to be tracked and updated after each forward pass through
  # a network.
  #
  # ## Arguments
  #
  # * l : `Array(Layer(T))` - Array of `Layer`s in the `Network`
  def build_params(l : Array(Layer(T)))
    l.each do |layer|
      layer.variables.each do |v|
        @params << v
        @first_moments << T.zeros_like(v.grad)
        @second_moments << T.zeros_like(v.grad)
      end
    end
  end

  # Updates all `Num::Grad::Variable`s registered to the optimizer
  # based on weights present in the network and the parameters of
  # the optimizer.  Resets all gradients to `0`.
  def update
    lr_t = @learning_rate * Math.sqrt(1 - @beta2_t) / (1 - @beta1_t)

    @beta1_t *= @beta1
    @beta2_t *= @beta2

    @params.size.times do |i|
      v = @params[i]

      if v.requires_grad
        @first_moments[i].map!(v.grad) do |x, y|
          @beta1 * x + (1 - @beta1) * y
        end

        @second_moments[i].map!(v.grad) do |x, y|
          @beta2 * x + (1 - @beta2) * y * y
        end

        v.value.map!(@first_moments[i], @second_moments[i]) do |x, y, z|
          x - lr_t * y / (Math.sqrt(z) + @epsilon)
        end

        v.grad = T.zeros_like(v.value)
      end
    end
  end
end
