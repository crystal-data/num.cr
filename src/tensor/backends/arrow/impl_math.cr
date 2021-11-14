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

  private macro elementwise_arrow(name, operator)
    # Implements the {{ operator }} operator between two `Tensor`s.
    # Broadcasting rules apply, the method is applied elementwise_arrow.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS to the operation
    # * b : `Tensor(U, ARROW(U))` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = [4, 5, 6].to_tensor
    # Num.{{ name }}(a, b)
    # ```
    def {{name}}(
      a : Tensor(U, ARROW(U)),
      b : Tensor(V, ARROW(V))
    ) : Tensor forall U, V
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
    # * a : `Tensor(U, ARROW(U))` - LHS to the operation
    # * b : `Tensor(U, ARROW(U))` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = [4, 5, 6].to_tensor
    # Num.{{ name }}!(a, b) # a is modified
    # ```
    def {{name}}!(
      a : Tensor(U, ARROW(U)),
      b : Tensor(V, ARROW(V))
    ) : Nil forall U, V
      a.map!(b) do |i, j|
        i {{operator.id}} j
      end
    end

    # Implements the {{ operator }} operator between a `Tensor` and scalar.
    # The scalar is broadcasted across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS to the operation
    # * b : `Number` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}(a, b)
    # ```
    def {{name}}(
      a : Tensor(U, ARROW(U)),
      b : Number
    ) : Tensor forall U
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
    # * a : `Tensor(U, ARROW(U))` - LHS to the operation
    # * b : `Number` - RHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}!(a, b)
    # ```
    def {{name}}!(a : Tensor(U, ARROW(U)), b : Number) : Nil forall U
      a.map! do |i|
        i {{operator.id}} b
      end
    end

    # Implements the {{ operator }} operator between a scalar and `Tensor`.
    # The scalar is broadcasted across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Number` - RHS to the operation
    # * b : `Tensor(U, ARROW(U))` - LHS to the operation
    #
    # ## Examples
    #
    # ```crystal
    # a = [1, 2, 3].to_tensor
    # b = 4
    # Num.{{ name }}(b, a)
    # ```
    def {{name}}(
      a : Number,
      b : Tensor(U, ARROW(U))
    ) : Tensor forall U
      b.map do |i|
        a {{operator.id}} i
      end
    end
  end

  # Implements the negation operator on a `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to negate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.negate(a) # => [-1, -2, -3]
  # ```
  def negate(a : Tensor(U, ARROW(U))) : Tensor(U, ARROW(U)) forall U
    a.map do |i|
      -i
    end
  end

  elementwise_arrow add, :+
  elementwise_arrow subtract, :-
  elementwise_arrow multiply, :*
  elementwise_arrow divide, :/
  elementwise_arrow floordiv, ://
  elementwise_arrow power, :**
  elementwise_arrow modulo, :%
  elementwise_arrow left_shift, :<<
  elementwise_arrow right_shift, :>>
  elementwise_arrow bitwise_and, :&
  elementwise_arrow bitwise_or, :|
  elementwise_arrow bitwise_xor, :^
  elementwise_arrow greater, :>
  elementwise_arrow greater_equal, :>=
  elementwise_arrow equal, :==
  elementwise_arrow not_equal, :!=
  elementwise_arrow less, :<
  elementwise_arrow less_equal, :<=

  private macro stdlibwrap1d_arrow(fn)
    # Implements the stdlib Math method {{ fn }} on a `Tensor`,
    # broadcasting the operation across all elements of the `Tensor`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - Argument to be operated upon
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # Num.{{ fn }}(a)
    # ```
    def {{fn.id}}(a : Tensor(U, ARROW(U))) : Tensor forall U
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
    # * a : `Tensor(U, ARROW(U))` - Argument to be operated upon
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # Num.{{ fn }}(a)
    # ```
    def {{fn.id}}!(a : Tensor(U, ARROW(U))) : Nil forall U
      a.map! do |i|
        Math.{{fn.id}}(i)
      end
    end
  end

  stdlibwrap1d_arrow acos
  stdlibwrap1d_arrow acosh
  stdlibwrap1d_arrow asin
  stdlibwrap1d_arrow asinh
  stdlibwrap1d_arrow atan
  stdlibwrap1d_arrow atanh
  stdlibwrap1d_arrow besselj0
  stdlibwrap1d_arrow besselj1
  stdlibwrap1d_arrow bessely0
  stdlibwrap1d_arrow bessely1
  stdlibwrap1d_arrow cbrt
  stdlibwrap1d_arrow cos
  stdlibwrap1d_arrow cosh
  stdlibwrap1d_arrow erf
  stdlibwrap1d_arrow erfc
  stdlibwrap1d_arrow exp
  stdlibwrap1d_arrow exp2
  stdlibwrap1d_arrow expm1
  stdlibwrap1d_arrow gamma
  stdlibwrap1d_arrow ilogb
  stdlibwrap1d_arrow lgamma
  stdlibwrap1d_arrow log
  stdlibwrap1d_arrow log10
  stdlibwrap1d_arrow log1p
  stdlibwrap1d_arrow log2
  stdlibwrap1d_arrow logb
  stdlibwrap1d_arrow sin
  stdlibwrap1d_arrow sinh
  stdlibwrap1d_arrow sqrt
  stdlibwrap1d_arrow tan
  stdlibwrap1d_arrow tanh

  private macro stdlibwrap_arrow(fn)
    # Implements the stdlib Math method {{ fn }} on two `Tensor`s,
    # broadcasting the `Tensor`s together.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS argument to the method
    # * b : `Tensor(V, ARROW(V))` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = [1.45, 3.2, 1.18]
    # Num.{{ fn }}(a, b)
    # ```
    def {{fn.id}}(
      a : Tensor(U, ARROW(U)),
      b : Tensor(V, ARROW(V))
    ) : Tensor forall U, V
      a.map(b) do |i, j|
        Math.{{fn.id}}(i, j)
      end
    end

    # Implements the stdlib Math method {{ fn }} on a `Tensor`,
    # broadcasting the `Tensor`s together.  The second `Tensor` must
    # broadcast against the shape of the first, as the first `Tensor`
    # is modified inplace.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS argument to the method
    # * b : `Tensor(V, ARROW(V))` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = [1.45, 3.2, 1.18]
    # Num.{{ fn }}!(a, b)
    # ```
    def {{fn.id}}!(
      a : Tensor(U, ARROW(U)),
      b : Tensor(V, ARROW(V))
    ) : Nil forall U, V
      a.map(b) do |i, j|
        Math.{{fn.id}}(i, j)
      end
    end

    # Implements the stdlib Math method {{ fn }} on a `Tensor` and a
    # Number, broadcasting the method across all elements of a `Tensor`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS argument to the method
    # * b : `Number` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = 1.5
    # Num.{{ fn }}(a, b)
    # ```
    def {{fn.id}}(
      a : Tensor(U, ARROW(U)),
      b : Number
    ) : Tensor forall U
      a.map do |i|
        Math.{{fn.id}}(i, b)
      end
    end

    # Implements the stdlib Math method {{ fn }} on a `Tensor` and a
    # Number, broadcasting the method across all elements of a `Tensor`.
    # The `Tensor` is modified inplace
    #
    # ## Arguments
    #
    # * a : `Tensor(U, ARROW(U))` - LHS argument to the method
    # * b : `Number` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = 1.5
    # Num.{{ fn }}!(a, b)
    # ```
    def {{fn.id}}!(a : Tensor(U, ARROW(U)), b : Number) : Nil forall U
      a.map! do |i|
        Math.{{fn.id}}(i, b)
      end
    end

    # Implements the stdlib Math method {{ fn }} on a `Number` and a
    # `Tensor`, broadcasting the method across all elements of a `Tensor`
    #
    # ## Arguments
    #
    # * a : `Number` - RHS argument to the method
    # * b : `Tensor(U, ARROW(U))` - LHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = 1.5
    # b = [2.0, 3.65, 3.141].to_tensor
    # Num.{{ fn }}(a, b)
    # ```
    def {{fn.id}}(
      a : Number,
      b : Tensor(U, ARROW(U))
    ) : Tensor forall U
      b.map do |i|
        Math.{{fn.id}}(a, i)
      end
    end
  end

  stdlibwrap_arrow atan2
  stdlibwrap_arrow besselj
  stdlibwrap_arrow bessely
  stdlibwrap_arrow copysign
  stdlibwrap_arrow hypot
  stdlibwrap_arrow ldexp
  stdlibwrap_arrow max
  stdlibwrap_arrow min
end
