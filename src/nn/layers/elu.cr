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

# Exponential Linear Unit or its widely known name ELU is a function
# that tend to converge cost to zero faster and produce more
# accurate results. Different to other activation functions,
# ELU has a extra alpha constant which should be positive number.
#
# ELU is very similiar to RELU except negative inputs. They are both
# in identity function form for non-negative inputs. On the other
# hand, ELU becomes smooth slowly until its output equal to -α
# whereas RELU sharply smoothes.
class Num::NN::EluLayer(T) < Num::NN::Layer(T)
  @alpha : Float32 | Float64 = 0.01
  getter output_shape : Array(Int32)

  # Initializes an ELU activation layer as part of a `Num::NN::Network`
  #
  # ## Arguments
  #
  # * context : `Num::Grad::Context(T)` - Context of the `Num::NN::Network`,
  #   used only to determine generic type of the `Num::NN::Layer(T)`
  # * output_shape : `Array(Int32)` - The shape of the output of the layer
  # * alpha : `Float` - Scale for the negative factor
  def initialize(
    context : Num::Grad::Context(T),
    @output_shape : Array(Int32),
    @alpha : Float32 | Float64 = 0.01
  )
  end

  # Computes a forward pass through an ELU layer.
  #
  # ## Arguments
  #
  # * input : `Num::Grad::Variable(T)` - Variable to activate
  def forward(input : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    output = Num::NN.elu(input.value, @alpha)
    result = input.context.variable(output)

    if input.is_grad_needed
      gate = Num::NN::EluGate.new(input.value)
      gate.cache(result, input)
    end
    result
  end
end

class Num::Grad::Variable(T)
  # Exponential Linear Unit activation function
  #
  # ## Arguments
  #
  # * alpha : `Float` - Scale for the negative factor
  def elu(alpha = 0.01)
    output = Num::NN.elu(@value, alpha)
    result = @context.variable(output)

    if self.is_grad_needed
      gate = Num::NN::EluGate.new(@value)
      gate.cache(result, self)
    end
    result
  end
end
