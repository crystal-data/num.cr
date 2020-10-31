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

class Num::Grad::Trig1dGate(T) < Num::Grad::Gate(T)
  getter a : Num::Grad::Variable(T)
  @@name = "Trig1d"

  def initialize(@a : Num::Grad::Variable(T))
  end

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    i * j
  end

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    gradient = payload.variable.grad
    r0 = gradient.map(a.value) do |i, j|
      self.derivative(i, j)
    end
    [r0]
  end

  def cache(result : Num::Grad::Variable(T), *args)
    a = args[0]
    result.grad = T.zeros_like(result.value)
    result.requires_grad = true
    Num::Grad.register(@@name, self, result, a)
  end
end

class Num::Grad::SinGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Sin"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    i * Math.cos(j)
  end
end

class Num::Grad::CosGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Cos"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    i * -Math.sin(j)
  end
end

class Num::Grad::TanGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Tan"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    i / Math.cos(j) ** 2
  end
end

class Num::Grad::ASinGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "ASin"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    if j.abs != 1
      i / Math.sqrt(1 - j ** 2)
    else
      U::NAN
    end
  end
end

class Num::Grad::ACosGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "ACos"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    if j.abs != 1
      -i / Math.sqrt(1 - j ** 2)
    else
      U::NAN
    end
  end
end

class Num::Grad::ATanGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Sin"

  @[AlwaysInline]
  def derivative(i : U, j : U) : U forall U
    i / (1 + j ** 2)
  end
end
