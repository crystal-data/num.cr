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
    # Implements the {{ operator }} operator between two `Tensor`s.
    # Broadcasting rules apply, the method is applied elementwise.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - LHS to the operation
    # * b : `Tensor(U, CPU(U))` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = [4, 5, 6].to_tensor
    # Num.{{ name }}(a, b)
    # ```
    @[AlwaysInline]
    def {{name}}(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map(b) do |i, j|
        i {{operator.id}} j
      end
    end

    # Implements the {{ operator }} operator between two `Tensor`s.
    # Broadcasting rules apply, the method is applied elementwise.
    # This method applies the operation inplace, storing the result
    # in the LHS argument.  Broadcasting cannot occur for the LHS
    # operand, so the second argument must broadcast to the first
    # operand's shape.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - LHS to the operation
    # * b : `Tensor(U, CPU(U))` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = [4, 5, 6].to_tensor
    # Num.{{ name }}!(a, b) # a is modified
    # ```
    @[AlwaysInline]
    def {{name}}!(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
      a.map!(b) do |i, j|
        i {{operator.id}} j
      end
    end

    # Implements the {{ operator }} operator between a `Tensor` and scalar.
    # The scalar is broadcasted across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - LHS to the operation
    # * b : `Number | Complex` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}(a, b)
    # ```
    @[AlwaysInline]
    def {{name}}(a : Tensor(U, CPU(U)), b : Number | Complex) forall U
      a.map do |i|
        i {{operator.id}} b
      end
    end

    # Implements the {{ operator }} operator between a `Tensor` and scalar.
    # The scalar is broadcasted across all elements of the `Tensor`, and the
    # `Tensor` is modified inplace.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - LHS to the operation
    # * b : `Number | Complex` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}!(a, b)
    # ```
    @[AlwaysInline]
    def {{name}}!(a : Tensor(U, CPU(U)), b : Number | Complex) forall U
      a.map! do |i|
        i {{operator.id}} b
      end
    end

    # Implements the {{ operator }} operator between a scalar and `Tensor`.
    # The scalar is broadcasted across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Number | Complex` - RHS to the operation
    # * b : `Tensor(U, CPU(U))` - LHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}(b, a)
    # ```
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

  elementwise add, :+
  elementwise subtract, :-
  elementwise multiply, :*
  elementwise divide, :/
  elementwise floordiv, ://
  elementwise power, :**
  elementwise modulo, :%
  elementwise left_shift, :<<
  elementwise right_shift, :>>
  elementwise bitwise_and, :&
  elementwise bitwise_or, :|
  elementwise bitwise_xor, :^

  private macro stdlibwrap1d(fn)
    # Implements the stdlib Math method {{ fn }} on a `Tensor`,
    # broadcasting the operation across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - Argument to be operated upon
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # Num.{{ fn }}(a)
    # ```
    @[AlwaysInline]
    def {{fn.id}}(a : Tensor(U, CPU(U))) forall U
      a.map do |i|
        Math.{{fn.id}}(i)
      end
    end

    # Implements the stdlib Math method {{ fn }} on a `Tensor`,
    # broadcasting the operation across all elements of the `Tensor`.
    # The `Tensor` is modified inplace to store the result
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - Argument to be operated upon
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # Num.{{ fn }}(a)
    # ```
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
