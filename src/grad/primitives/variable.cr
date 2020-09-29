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
  # The graph the variable is associated with.  This is a reference,
  # as a variable does not own its context
  getter context : Num::Grad::Context(T)

  # The value of the Variable.  This should not be edited outside
  # of Variable operations, as other edits will not be tracked
  # and will lead to incorrect results
  getter value : T

  # The gradient of the Variable.  This is set as a reference to
  # the value of a Variable unless `backprop` has been called, in
  # which case all related Variables will have their gradient
  # updated correctly
  property grad : T

  # If set to true, this variable will track its operations,
  # otherwise it will act similar to a Tensor, only calculating
  # forward operations
  property requires_grad : Bool

  # Initialization method for a Variable.
  #
  # This method should only be called by a context, as it creates
  # a Variable.  Context provides a helper method to add a
  # Variable to the computational graph that handles ownership
  # of the context and other related instance variables
  def initialize(
    @context : Num::Grad::Context(T),
    @value : T,
    @requires_grad : Bool = false
  )
    if @requires_grad
      @grad = T.zeros_like(@value)
    else
      @grad = @value
    end
  end

  # :nodoc:
  def is_grad_needed : Bool
    @requires_grad && !@context.no_grad
  end

  # :nodoc:
  def to_s(io)
    @value.to_s(io)
  end

  # Back propogates an operation along a computational graph.
  # This operation will destroy the operational graph, populating
  # the gradients for all variables that are predecessors of
  # the Variable this is called on.
  #
  # Even if this is called on the first node in a graph, it will
  # destroy all descendents of this variable stored by the
  # Context
  def backprop
    @grad = T.ones_like(@value)

    while @context.size > 0 && @context.last.payload.variable != self
      @context.pop
    end

    while @context.size > 0
      cur_node = @context.pop
      diffs = cur_node.gate.backward(cur_node.payload)
      diffs.each_with_index do |diff, i|
        parent_i = cur_node.parents[i]
        if parent_i.requires_grad
          parent_i.grad += diff
        end
      end
    end
  end
end
