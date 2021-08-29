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

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    [] of T
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

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.sin_backward(payload.variable.grad, @a)
  end
end

class Num::Grad::CosGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Cos"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.cos_backward(payload.variable.grad, @a)
  end
end

class Num::Grad::TanGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Tan"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.tan_backward(payload.variable.grad, @a)
  end
end

class Num::Grad::ASinGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "ASin"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.asin_backward(payload.variable.grad, @a)
  end
end

class Num::Grad::ACosGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "ACos"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.acos_backward(payload.variable.grad, @a)
  end
end

class Num::Grad::ATanGate(T) < Num::Grad::Trig1dGate(T)
  @@name = "Sin"

  def backward(payload : Num::Grad::Payload(T)) : Array(T)
    Num::Grad.atan_backward(payload.variable.grad, @a)
  end
end
