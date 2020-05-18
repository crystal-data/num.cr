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

require "../tensor/tensor"
require "math"

module Num
  extend self

  macro elementwise(name, operator)
    def {{name}}(a : Tensor, b : Tensor)
      a.map2(b) { |i, j| i {{operator.id}} j }
    end

    def {{name}}!(a : Tensor, b : Tensor)
      a.map2!(b) { |i, j| i {{operator.id}} j }
    end

    def {{name}}(a : Tensor, b : Number)
      a.map { |i| i {{operator.id}} b }
    end

    def {{name}}!(a : Tensor, b : Number)
      a.map! { |i| i {{operator.id}} b }
    end

    def {{name}}(a : Number, b : Tensor)
      b.map { |i| a {{operator.id}} i }
    end

    def {{name}}(a : Number, b : Number)
      a {{operator.id}} b
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
  elementwise equal, :==
  elementwise not_equal, :!=
  elementwise greater, :>
  elementwise greater_equal, :>=
  elementwise less, :<
  elementwise less_equal, :<=

  macro stdlibwrap1d(fn)
    def {{fn.id}}(a : Tensor)
      a.map { |i| Math.{{fn.id}}(i) }
    end

    def {{fn.id}}!(a : Tensor)
      a.map! { |i| Math.{{fn.id}}(i) }
    end

    def {{fn.id}}(a : Number)
      Math.{{fn.id}}(a)
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

  macro stdlibwrap(fn)
    def {{fn.id}}(a : Tensor, b : Tensor)
      a.map2(b) { |i, j| Math.{{fn.id}}(i, j) }
    end

    def {{fn.id}}!(a : Tensor, b : Tensor)
      a.map2!(b) { |i, j| Math.{{fn.id}}(i, j) }
    end

    def {{fn.id}}(a : Tensor, b : Number)
      a.map { |i| Math.{{fn.id}}(i, b) }
    end

    def {{fn.id}}!(a : Tensor, b : Number)
      a.map! { |i| Math.{{fn.id}}(i, b) }
    end

    def {{fn.id}}(a : Number, b : Tensor)
      b.map { |i| Math.{{fn.id}}(a, i) }
    end

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
end
