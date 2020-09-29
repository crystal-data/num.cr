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

# A variable is an abstraction of a Tensor that tracks
# the operations done to the Tensor.  It also keeps
# track of the gradient of the operation if a Variable
# needs to backpropogate.
#
# This is the fundamental object used in automatic
# differentiation, as well as the neural network aspects
# of Num.cr
class Num::Grad::Variable(T)
  private macro operator_op(operator, gate_cls, *args)
    def {{operator.id}}(other : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
      result = @context.variable(@value {{operator.id}} other.value)

      if self.is_grad_needed || other.is_grad_needed
        gate = {{gate_cls}}.new {{*args}}
        gate.cache(result, self, other)
      end

      result
    end
  end

  # Adds a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a + b # => [5.0]
  # f.backprop
  # ```
  operator_op :+, Num::Grad::AddGate(T)

  # Subtracts a variable from another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a - b # => [-1.0]
  # f.backprop
  # ```
  operator_op :-, Num::Grad::SubtractGate(T)

  # Multiples a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a * b # => [6.0]
  # f.backprop
  # ```
  operator_op :*, Num::Grad::MultiplyGate(T), self, other

  # Raises a variable to another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a ** b # => [8.0]
  # f.backprop
  # ```
  operator_op :**, Num::Grad::PowerGate(T), self, other

  # Divides a variable by another variable and stores
  # the derivative of the operation in the computational
  # graph.
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([2.0])
  # b = ctx.variable([3.0])
  #
  # f = a / b # => [0.66667]
  # f.backprop
  # ```
  operator_op :/, Num::Grad::DivideGate(T), self, other

  # Matrix multiply operator for two variables.  Computes the
  # dot product of two matrices and stores the result in the
  # computational graph
  #
  # Arguments
  # ---------
  # *other* : Num::Grad::Variable(T)
  #   - right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([[2.0], [2.0]])
  # b = ctx.variable([[3.0, 3.0]])
  #
  # f = a + b
  #
  # # [[6, 6],
  # #  [6, 6]]
  #
  # f.backprop
  # ```
  def matmul(b : Num::Grad::Variable(T)) : Num::Grad::Variable(T)
    result = Num::Grad::Variable.new(@context, self.value.matmul(b.value))

    if self.is_grad_needed || b.is_grad_needed
      gate = Num::Grad::MatMulGate(T).new(self, b)
      gate.cache(result, self, b)
    end
    result
  end

  # Slices a variable.  Slices the gradient of the variable
  # using the same arguments
  #
  # Arguments
  # ---------
  # *args*
  #   Slicing arguments, slicing behavior is the same as
  #   it is for a standard Tensor
  #
  # Examples
  # --------
  # ```
  # ctx = Num::Grad::Context(Tensor(Float64)).new
  #
  # a = ctx.variable([[2.0], [3.0]])
  # b = a[1]
  # b # => [3]
  # ```
  def [](*args)
    output = @value[*args]
    result = @context.variable(output, requires_grad: @requires_grad)
    result.grad = @grad[*args]
    result
  end
end
