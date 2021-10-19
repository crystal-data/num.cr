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

    # :ditto:
    def {{op.id}}(other)
      Num.{{name.id}}(self, other)
    end
  end

  # Adds two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a + a
  # ```
  alias_to_backend add, :+

  # Subtracts two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a - a
  # ```
  alias_to_backend subtract, :-

  # Multiplies two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a * a
  # ```
  alias_to_backend multiply, :*

  # Divides two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a / a
  # ```
  alias_to_backend divide, :/

  # Floor divides two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a // a
  # ```
  alias_to_backend floordiv, ://

  # Exponentiates two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a ** a
  # ```
  alias_to_backend power, :**

  # Return element-wise remainder of division for two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a % a
  # ```
  alias_to_backend modulo, :%

  # Shift the bits of an integer to the left.
  # Bits are shifted to the left by appending x2 0s at the right of x1.
  # Since the internal representation of numbers is in binary format,
  # this operation is equivalent to multiplying x1 by 2**x2.
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a << a
  # ```
  alias_to_backend left_shift, :<<

  # Shift the bits of an integer to the right.
  #
  # Bits are shifted to the right x2. Because the internal representation
  # of numbers is in binary format, this operation is equivalent to
  # dividing x1 by 2**x2.
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a >> a
  # ```
  alias_to_backend right_shift, :>>

  # Compute the bit-wise AND of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a & a
  # ```
  alias_to_backend bitwise_and, :&

  # Compute the bit-wise OR of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a | a
  # ```
  alias_to_backend bitwise_or, :|

  # Compute the bit-wise XOR of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *other* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a ^ a
  # ```
  alias_to_backend bitwise_xor, :^

  private macro delegate_to_backend(name)
    def {{name.id}}
      Num.{{name.id}}(self)
    end
  end

  # Trigonometric inverse cosine, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.acos
  # ```
  delegate_to_backend acos

  # Inverse hyperbolic cosine, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.acos
  # ```
  delegate_to_backend acosh

  # Inverse sine, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.asin
  # ```
  delegate_to_backend asin

  # Inverse hyperbolic sine, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.asinh
  # ```
  delegate_to_backend asinh

  # Inverse tangent, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.atan
  # ```
  delegate_to_backend atan

  # Inverse hyperbolic tangent, element-wise.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.atanh
  # ```
  delegate_to_backend atanh

  # Calculates besselj0, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.besselj0
  # ```
  delegate_to_backend besselj0

  # Calculates besselj1, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.besselj1
  # ```
  delegate_to_backend besselj1

  # Calculates bessely0, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.bessely0
  # ```
  delegate_to_backend bessely0

  # Calculates bessely1, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.bessely1
  # ```
  delegate_to_backend bessely1

  # Calculates cube root, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cbrt
  # ```
  delegate_to_backend cbrt

  # Calculates cosine, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cos
  # ```
  delegate_to_backend cos

  # Calculates hyperbolic cosine, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cosh
  # ```
  delegate_to_backend cosh

  # Calculates erf, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.erf
  # ```
  delegate_to_backend erf

  # Calculates erfc, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.erfc
  # ```
  delegate_to_backend erfc

  # Calculates exp, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.exp
  # ```
  delegate_to_backend exp

  # Calculates exp2, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.exp2
  # ```
  delegate_to_backend exp2

  # Calculates expm1, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.expm1
  # ```
  delegate_to_backend expm1

  # Calculates gamma function, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.gamma
  # ```
  delegate_to_backend gamma

  # Calculates ilogb, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.ilogb
  # ```
  delegate_to_backend ilogb

  # Calculates logarithmic gamma, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.lgamma
  # ```
  delegate_to_backend lgamma

  # Calculates log, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log
  # ```
  delegate_to_backend log

  # Calculates log10, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log10
  # ```
  delegate_to_backend log10

  # Calculates log1p, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log1p
  # ```
  delegate_to_backend log1p

  # Calculates log2, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log2
  # ```
  delegate_to_backend log2

  # Calculates logb, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.logb
  # ```
  delegate_to_backend logb

  # Calculates sine, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sin
  # ```
  delegate_to_backend sin

  # Calculates hyperbolic sine, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sinh
  # ```
  delegate_to_backend sinh

  # Calculates square root, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sqrt
  # ```
  delegate_to_backend sqrt

  # Calculates tangent, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.tan
  # ```
  delegate_to_backend tan

  # Calculates hyperbolic tangent, elementwise
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.tanh
  # ```
  delegate_to_backend tanh
end
