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
class Num::Grad::AddGate(T) < Num::Grad::Gate(T)
  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad
    [gradient, gradient]
  end

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args

    result.grad = T.zeros_like(result.value)
    result.requires_grad = true

    Num::Grad.register("Add", self, result, a, b)
  end
end

# :nodoc:
class Num::Grad::SubtractGate(T) < Num::Grad::Gate(T)
  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad
    [gradient, -gradient]
  end

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true

    Num::Grad.register("Sub", self, result, a, b)
  end
end

# :nodoc:
class Num::Grad::MultiplyGate(T) < Num::Grad::Gate(T)
  getter a : Num::Grad::Variable(T)
  getter b : Num::Grad::Variable(T)

  # :nodoc:
  def initialize(@a : Num::Grad::Variable(T), @b : Num::Grad::Variable(T))
  end

  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad

    [gradient * @b.value, @a.value * gradient]
  end

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true

    Num::Grad.register("Mul", self, result, a, b)
  end
end

# :nodoc:
class Num::Grad::DivideGate(T) < Num::Grad::Gate(T)
  getter a : Num::Grad::Variable(T)
  getter b : Num::Grad::Variable(T)

  # :nodoc:
  def initialize(@a : Num::Grad::Variable(T), @b : Num::Grad::Variable(T))
  end

  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad

    r0 = gradient.map(@b.value) { |i, j| i / j }
    r1 = gradient.map(@a.value, @b.value) { |i, j, k| -i * j / (k ** 2) }
    [r0, r1]
  end

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true
    Num::Grad.register("Div", self, result, a, b)
  end
end

# :nodoc:
class Num::Grad::PowerGate(T) < Num::Grad::Gate(T)
  getter a : Num::Grad::Variable(T)
  getter b : Num::Grad::Variable(T)

  # :nodoc:
  def initialize(@a : Num::Grad::Variable(T), @b : Num::Grad::Variable(T))
  end

  # :nodoc:
  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad

    r0 = gradient.map(a.value, b.value) do |grad, x, y|
      grad * y * (x ** (y == 0 ? 1 : y - 1))
    end

    r1 = gradient.map(a.value, b.value) do |grad, x, y|
      grad * (x ** y) * Math.log(x == 0 ? 1 : x)
    end

    [r0, r1]
  end

  # :nodoc:
  def cache(result : Num::Grad::Variable(T), *args)
    a, b = args
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true
    Num::Grad.register("Pow", self, result, a, b)
  end
end
