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
require "../api"

module Num::NN
  extend self

  class Context(T)
    @nodes : Array(Node(T))
    @no_grad : Bool

    def initialize(@no_grad = false)
      @nodes = Array(Node(T)).new
    end

    def size
      @nodes.size
    end

    def <<(node : Node(T))
      @nodes << node
    end

    def last
      @nodes.last
    end

    def pop
      @nodes.pop
    end

    def no_grad_mode
      prev_state = @no_grad
      @no_grad = true
      yield
      @no_grad = prev_state
    end
  end

  class Variable(T)
    getter context : Context(T)
    getter value : T
    getter grad : T? = nil
    @requires_grad : Bool

    def initialize(@context, @value, @requires_grad = false)
      if @requires_grad
        @grad = @value.zeros_like
      end
    end

    def is_grad_needed
      @requires_grad && @context.no_grad
    end

    def backward
      @grad = @value.ones_like
      while @context.size > 0 && @context.peek.payload.variable != v
        @context.pop
      end

      while @context.size > 0
        node = @context.pop
        diffs = @node.gate.backward(node.payload)
        diffs.each_with_index do |diff, i|
          parent_i = nodes.parents[i]
          if parent_i.requires_grad
            Num.add!(parent_i.grad, diff)
          end
        end
      end
    end

    def add_cache(a : Variable(T), b : Variable(T))
      gate = AddGate(T).new
      @grad = @value.zeros_like
      @requires_grad = true
      register_node(
        "Add",
        gate,
        self,
        a, b
      )
    end

    def +(other : Variable(T))
      val = Num.add(self.value, other.value)
      result = Variable(T).new(@context, val)

      if is_grad_needed || other.is_grad_needed
        result.add_cache(self, other)
      end
      result
    end

    def sub_cache(a : Variable(T), b : Variable(T))
      gate = SubGate(T).new
      @grad = @value.zeros_like
      @requires_grad = true

      register_node(
        "Sub",
        gate,
        self,
        a, b
      )
    end

    def -(other : Variable(T))
      val = Num.subtract(self.value, other.value)
      result = Variable(T).new(@context, val)

      if is_grad_needed || other.is_grad_needed
        result.sub_cache(self, other)
      end
      result
    end

    def matmul_cache(a : Variable(T), b : Variable(T))
      gate = MatMulGate(T).new(a, b)

      @grad = @value.zeros_like
      @requires_grad = true

      register_node(
        "MatMul",
        gate,
        self,
        a, b
      )
    end

    def *(other : Variable(T))
      val = @value.matmul(other.value)
      result = Variable(T).new(@context, val)

      if is_grad_needed || other.is_grad_needed
        result.matmul_cache(self, other)
      end
      result
    end
  end

  class Gate(T)
    def backward(payload : Payload(T))
    end
  end

  class AddGate(T) < Gate(T)
    def backward(payload : Payload(T))
      gradient = payload.variable.grad
      [gradient, gradient]
    end
  end

  class SubGate(T) < Gate(T)
    def backward(payload : Payload(T))
      gradient = payload.variable.grad
      [gradient, -gradient]
    end
  end

  class MatMulGate(T) < Gate(T)
    @a : Variable(T)
    @b : Variable(T)

    def initialize(@a, @b)
    end

    def backward(payload : Payload(T))
      gradient = payload.variable.grad
      [
        gradient.matmul(@b.value.transpose),
        @a.value.transpose.matmul(gradient)
      ]
    end
  end

  enum PayloadType
    Var
  end

  class Payload(T)
    @kind : PayloadType
    getter variable : Variable(T)

    def initialize(@kind, @variable)
    end
  end

  class Node(T)
    getter gate : Gate(T)
    @parents : Array(Variable(T))
    getter payload : Payload(T)
    @name : String

    def initialize(@gate, @parents, @payload, @name = "")
    end
  end

  def new_context(typedesc : U.class) forall U
    Context(U).new
  end

  def variable(ctx : Context(U), value : U, requires_grad = false) forall U
    Variable(U).new(ctx, value, requires_grad)
  end

  def register_node(name : String, gate : Gate(U), result : Variable(U), *parents : Variable(U)) forall U
    payload = Payload(U).new(result)
    node = Node(U).new(gate, parents.to_a, payload, name)
    parents[0].context << node
  end
end
