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

class Tensor(T, S)
  private macro alias_to_backend(name, op)
    def {{name.id}}(other)
      Num.{{name.id}}(self, other)
    end
    def {{op.id}}(other)
      Num.{{name.id}}(self, other)
    end
  end

  alias_to_backend add, :+
  alias_to_backend subtract, :-
  alias_to_backend multiply, :*
  alias_to_backend divide, :/
  alias_to_backend floordiv, ://
  alias_to_backend power, :**
  alias_to_backend modulo, :%
  alias_to_backend left_shift, :<<
  alias_to_backend right_shift, :>>
  alias_to_backend bitwise_and, :&
  alias_to_backend bitwise_or, :|
  alias_to_backend bitwise_xor, :^

  private macro delegate_to_backend(name)
    def {{name.id}}
      Num.{{name.id}}(self)
    end
  end

  delegate_to_backend acos
  delegate_to_backend acosh
  delegate_to_backend asin
  delegate_to_backend asinh
  delegate_to_backend atan
  delegate_to_backend atanh
  delegate_to_backend besselj0
  delegate_to_backend besselj1
  delegate_to_backend bessely0
  delegate_to_backend bessely1
  delegate_to_backend cbrt
  delegate_to_backend cos
  delegate_to_backend cosh
  delegate_to_backend erf
  delegate_to_backend erfc
  delegate_to_backend exp
  delegate_to_backend exp2
  delegate_to_backend expm1
  delegate_to_backend gamma
  delegate_to_backend ilogb
  delegate_to_backend lgamma
  delegate_to_backend log
  delegate_to_backend log10
  delegate_to_backend log1p
  delegate_to_backend log2
  delegate_to_backend logb
  delegate_to_backend sin
  delegate_to_backend sinh
  delegate_to_backend sqrt
  delegate_to_backend tan
  delegate_to_backend tanh
end
