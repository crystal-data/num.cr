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

struct Float32
  macro add_operator(name, operator)
    def {{operator.id}}(other : Num::Grad::Variable(Tensor(Float32)))
      other.context.variable(self) {{operator.id}} other
    end
  end

  add_operator add, :+
  add_operator subtract, :-
  add_operator multiply, :*
  add_operator divide, :/
  add_operator power, :**
end

struct Float64
  macro add_operator(name, operator)
    def {{operator.id}}(other : Num::Grad::Variable(Tensor(Float64)))
      other.context.variable(self) {{operator.id}} other
    end
  end

  add_operator add, :+
  add_operator subtract, :-
  add_operator multiply, :*
  add_operator divide, :/
  add_operator power, :**

  def exp
    Math.exp(self)
  end
end
