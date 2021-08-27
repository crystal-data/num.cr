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

# A Context keeps track of the computational graph for
# a number of operations.  Variables that interact with each
# other must belong to the same context, or state will be
# lost while tracking operations done.
#
# The generic type of a context is always going to be a specific
# class of `Tensor`, to allow easy creation of gradients on the
# fly.  Unlike standard `Tensor` operations, a `Context` cannot
# shift it's generic type, and operations resulting in a different
# data type will raise.

class Num::Grad::Context(T)
  # A list of all variables present in an operation.
  # This list can contain duplicates
  getter nodes : Array(Num::Grad::Node(T))

  # If no_grad is set to true, operations will not
  # be cached, and backpropogation will not be possible
  getter no_grad : Bool

  # Contexts can only be initialized as empty, and
  # a generic type must be provided
  def initialize
    @nodes = Array(Num::Grad::Node(T)).new
    @no_grad = false
  end

  # :nodoc:
  def size : Int32
    @nodes.size
  end

  # :nodoc:
  def <<(node : Num::Grad::Node(T))
    @nodes << node
  end

  def last : Num::Grad::Node(T)
    # :nodoc:
    @nodes.last
  end

  # :nodoc:
  def pop : Num::Grad::Node(T)
    @nodes.pop
  end

  # Creates a new variable within the `Context`.  This variable
  # must be able to be cast to a `Tensor` of type `T`.
  #
  # Arguments
  # ---------
  # *value* : Tensor-like
  #   A value that can be converted to a `Tensor`
  # *requires_grad* : Bool
  #   Flag to indicate if operations should be cached for this
  #   variable
  #
  # Examples
  # --------
  # ```
  # ctx = Context(Tensor(Float64)).new
  # ctx.variable([1.0, 2.0, 3.0])
  # ```
  def variable(value : T, requires_grad : Bool = true) : Num::Grad::Variable(T)
    Num::Grad::Variable.new(self, value, requires_grad)
  end

  def variable(value : Number, requires_grad : Bool = true) : Num::Grad::Variable(T)
    Num::Grad::Variable.new(self, T.new(value), requires_grad)
  end

  def variable(value : Array, requires_grad : Bool = true) : Num::Grad::Variable(T)
    Num::Grad::Variable.new(self, value.to_tensor, requires_grad)
  end

  # :nodoc:
  def to_s(io)
    @nodes.each_with_index do |node, i|
      io << "Node #{i}: #{node.name} - "
      if node.parents.size <= 1
        io << node.parents[0].value.shape
      else
        io << "("
        node.parents.each_with_index do |parent, pi|
          if pi != 0
            io << ", "
          end
          io << parent.value.shape
        end
        io << ")"
      end
      io << node.payload.variable.value.shape
      unless i == @nodes.size - 1
        io << "\n"
      end
    end
  end
end

module Num::Grad
  extend self

  # Cached a node in the computational graph.  This is only required
  # if an operation needs to be backpropogated.
  #
  # Arguments
  # ---------
  # *name* : String
  #   Description of the operation
  # *gate* : Gate(U)
  #   Operation gate containing a backward method and
  #   cached arguments
  # *result* : Variable(U)
  #   The result of the operation being cached
  # *parents* : *Variable(U)
  #   The operands present in operation being cached
  #
  # This method should be used sparingly by Users of the application.
  # It should only be necessary when a User is defining their own
  # custom activation function or Layer.
  def register(
    name : String,
    gate : Num::Grad::Gate(U),
    result : Num::Grad::Variable(U),
    *parents : Num::Grad::Variable(U)
  ) forall U
    payload = Num::Grad::Payload.new(result)
    node = Num::Grad::Node.new(gate, parents.to_a, payload, name)
    parents[0].context << node
  end
end
