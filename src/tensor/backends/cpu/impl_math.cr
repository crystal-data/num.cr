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

module Num
  extend self

  private macro elementwise(name, operator)
    @[AlwaysInline]
    def {{name}}(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map(b) do |i, j|
        i {{operator.id}} j
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{name}}!(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map!(b) do |i, j|
        i {{operator.id}} j
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{name}}(a : Tensor(U, CPU(U)), b : Number | Complex) forall U
      a.map do |i|
        i {{operator.id}} b
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{name}}!(a : Tensor(U, CPU(U)), b : Number | Complex) forall U
      a.map! do |i|
        i {{operator.id}} b
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{name}}(a : Number | Complex, b : Tensor(U, CPU(U))) forall U
      b.map do |i|
        a {{operator.id}} i
      end
    end
  end

  def negate(a : Tensor(U, CPU(U))) forall U
    a.map do |i|
      -i
    end
  end

  # Adds two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a + a
  # ```
  elementwise add, :+

  # Subtracts two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a - a
  # ```
  elementwise subtract, :-

  # Multiplies two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a * a
  # ```
  elementwise multiply, :*

  # Divides two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a / a
  # ```
  elementwise divide, :/

  # Floor divides two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a // a
  # ```
  elementwise floordiv, ://

  # Exponentiates two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a ** a
  # ```
  elementwise power, :**

  # Return element-wise remainder of division for two `Tensor`s elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1.5, 2.2, 3.2]
  # a % a
  # ```
  elementwise modulo, :%

  # Shift the bits of an integer to the left.
  # Bits are shifted to the left by appending x2 0s at the right of x1.
  # Since the internal representation of numbers is in binary format,
  # this operation is equivalent to multiplying x1 by 2**x2.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a << a
  # ```
  elementwise left_shift, :<<

  # Shift the bits of an integer to the right.
  #
  # Bits are shifted to the right x2. Because the internal representation
  # of numbers is in binary format, this operation is equivalent to
  # dividing x1 by 2**x2.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a >> a
  # ```
  elementwise right_shift, :>>

  # Compute the bit-wise AND of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a & a
  # ```
  elementwise bitwise_and, :&

  # Compute the bit-wise OR of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a | a
  # ```
  elementwise bitwise_or, :|

  # Compute the bit-wise XOR of two `Tensor`s element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   LHS argument
  # *b* : Tensor | Number
  #   RHS argument
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a ^ a
  # ```
  elementwise bitwise_xor, :^

  private macro stdlibwrap1d(fn)
    @[AlwaysInline]
    def {{fn.id}}(a : Tensor(U, CPU(U))) forall U
      a.map do |i|
        Math.{{fn.id}}(i)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}!(a : Tensor(U, CPU(U))) forall U
      a.map! do |i|
        Math.{{fn.id}}(i)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}(a : Number | Complex)
      Math.{{fn.id}}(a)
    end
  end

  # Trigonometric inverse cosine, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.acos
  # ```
  stdlibwrap1d acos

  # Inverse hyperbolic cosine, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.acos
  # ```
  stdlibwrap1d acosh

  # Inverse sine, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.asin
  # ```
  stdlibwrap1d asin

  # Inverse hyperbolic sine, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.asinh
  # ```
  stdlibwrap1d asinh

  # Inverse tangent, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.atan
  # ```
  stdlibwrap1d atan

  # Inverse hyperbolic tangent, element-wise.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.atanh
  # ```
  stdlibwrap1d atanh

  # Calculates besselj0, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.besselj0
  # ```
  stdlibwrap1d besselj0

  # Calculates besselj1, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.besselj1
  # ```
  stdlibwrap1d besselj1

  # Calculates bessely0, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.bessely0
  # ```
  stdlibwrap1d bessely0

  # Calculates bessely1, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.bessely1
  # ```
  stdlibwrap1d bessely1

  # Calculates cube root, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cbrt
  # ```
  stdlibwrap1d cbrt

  # Calculates cosine, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cos
  # ```
  stdlibwrap1d cos

  # Calculates hyperbolic cosine, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.cosh
  # ```
  stdlibwrap1d cosh

  # Calculates erf, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.erf
  # ```
  stdlibwrap1d erf

  # Calculates erfc, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.erfc
  # ```
  stdlibwrap1d erfc

  # Calculates exp, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.exp
  # ```
  stdlibwrap1d exp

  # Calculates exp2, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.exp2
  # ```
  stdlibwrap1d exp2

  # Calculates expm1, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.expm1
  # ```
  stdlibwrap1d expm1

  # Calculates gamma function, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.gamma
  # ```
  stdlibwrap1d gamma

  # Calculates ilogb, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.ilogb
  # ```
  stdlibwrap1d ilogb

  # Calculates logarithmic gamma, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.lgamma
  # ```
  stdlibwrap1d lgamma

  # Calculates log, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log
  # ```
  stdlibwrap1d log

  # Calculates log10, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log10
  # ```
  stdlibwrap1d log10

  # Calculates log1p, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log1p
  # ```
  stdlibwrap1d log1p

  # Calculates log2, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.log2
  # ```
  stdlibwrap1d log2

  # Calculates logb, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.logb
  # ```
  stdlibwrap1d logb

  # Calculates sine, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sin
  # ```
  stdlibwrap1d sin

  # Calculates hyperbolic sine, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sinh
  # ```
  stdlibwrap1d sinh

  # Calculates square root, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.sqrt
  # ```
  stdlibwrap1d sqrt

  # Calculates tangent, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.tan
  # ```
  stdlibwrap1d tan

  # Calculates hyperbolic tangent, elementwise
  #
  # Arguments
  # ---------
  # *a* : Tensor | Number
  #   Argument to operate on
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.tanh
  # ```
  stdlibwrap1d tanh

  private macro stdlibwrap(fn)
    @[AlwaysInline]
    def {{fn.id}}(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map(b) do |i, j|
        Math.{{fn.id}}(i, j)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}!(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map(b) do |i, j|
        Math.{{fn.id}}(i, j)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}(a : Tensor(U, CPU(U)), b : Number) forall U
      a.map do |i|
        Math.{{fn.id}}(i, b)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}!(a : Tensor(U, CPU(U)), b : Number) forall U
      a.map! do |i|
        Math.{{fn.id}}(i, b)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}(a : Number, b : Tensor(U, CPU(U))) forall U
      b.map do |i|
        Math.{{fn.id}}(a, i)
      end
    end

    # :ditto:
    @[AlwaysInline]
    def {{fn.id}}(a : Number, b : Number)
      Math.{{fn.id}}(a, b)
    end
  end

  stdlibwrap atan2
  stdlibwrap besselj
  stdlibwrap bessely
  stdlibwrap copysign
  stdlibwrap hypot
  stdlibwrap ldexp
  stdlibwrap max
  stdlibwrap min
end
