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
    
    def {{name}}(
      a : Tensor(U, CPU(U)),
      b : Tensor(V, CPU(V))
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
    
    def {{name}}!(
      a : Tensor(U, CPU(U)),
      b : Tensor(V, CPU(V))
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
    
    def {{name}}(
      a : Tensor(U, CPU(U)),
      b : Number | Complex
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
    
    def {{name}}!(a : Tensor(U, CPU(U)), b : Number | Complex) : Nil forall U
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
    
    def {{name}}(
      a : Number | Complex,
      b : Tensor(U, CPU(U))
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
  # * a : `Tensor(U, CPU(U))` - `Tensor` to negate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.negate(a) # => [-1, -2, -3]
  # ```
  def negate(a : Tensor(U, CPU(U))) : Tensor(U, CPU(U)) forall U
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
  elementwise greater, :>
  elementwise greater_equal, :>=
  elementwise equal, :==
  elementwise not_equal, :!=
  elementwise less, :<
  elementwise less_equal, :<=

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
    
    def {{fn.id}}(a : Tensor(U, CPU(U))) : Tensor forall U
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
    
    def {{fn.id}}!(a : Tensor(U, CPU(U))) : Nil forall U
      a.map! do |i|
        Math.{{fn.id}}(i)
      end
    end
  end

  stdlibwrap1d acos
  stdlibwrap1d acosh
  stdlibwrap1d asin
  stdlibwrap1d asinh
  stdlibwrap1d atan
  stdlibwrap1d atanh
  stdlibwrap1d besselj0
  stdlibwrap1d besselj1
  stdlibwrap1d bessely0
  stdlibwrap1d bessely1
  stdlibwrap1d cbrt
  stdlibwrap1d cos
  stdlibwrap1d cosh
  stdlibwrap1d erf
  stdlibwrap1d erfc
  stdlibwrap1d exp
  stdlibwrap1d exp2
  stdlibwrap1d expm1
  stdlibwrap1d gamma
  stdlibwrap1d ilogb
  stdlibwrap1d lgamma
  stdlibwrap1d log
  stdlibwrap1d log10
  stdlibwrap1d log1p
  stdlibwrap1d log2
  stdlibwrap1d logb
  stdlibwrap1d sin
  stdlibwrap1d sinh
  stdlibwrap1d sqrt
  stdlibwrap1d tan
  stdlibwrap1d tanh

  private macro stdlibwrap(fn)
    # Implements the stdlib Math method {{ fn }} on two `Tensor`s,
    # broadcasting the `Tensor`s together.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, CPU(U))` - LHS argument to the method
    # * b : `Tensor(V, CPU(V))` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = [1.45, 3.2, 1.18]
    # Num.{{ fn }}(a, b)
    # ```
    
    def {{fn.id}}(
      a : Tensor(U, CPU(U)),
      b : Tensor(V, CPU(V))
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
    # * a : `Tensor(U, CPU(U))` - LHS argument to the method
    # * b : `Tensor(V, CPU(V))` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = [1.45, 3.2, 1.18]
    # Num.{{ fn }}!(a, b)
    # ```
    
    def {{fn.id}}!(
      a : Tensor(U, CPU(U)),
      b : Tensor(V, CPU(V))
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
    # * a : `Tensor(U, CPU(U))` - LHS argument to the method
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
      a : Tensor(U, CPU(U)),
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
    # * a : `Tensor(U, CPU(U))` - LHS argument to the method
    # * b : `Number` - RHS argument to the method
    #
    # ## Examples
    #
    # ```crystal
    # a = [2.0, 3.65, 3.141].to_tensor
    # b = 1.5
    # Num.{{ fn }}!(a, b)
    # ```
    
    def {{fn.id}}!(a : Tensor(U, CPU(U)), b : Number) : Nil forall U
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
    # * b : `Tensor(U, CPU(U))` - LHS argument to the method
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
      b : Tensor(U, CPU(U))
    ) : Tensor forall U
      b.map do |i|
        Math.{{fn.id}}(a, i)
      end
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
