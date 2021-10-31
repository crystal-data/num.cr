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
  macro operator_op(fn, operator)
    {% for dtype in [Int32, UInt32, Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::ArithmeticKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}Inplace < Num::ArithmeticInplaceKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end
    {% end %}

    # {{ fn.stringify.capitalize.id }} two `Tensor`s elementwise
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to {{ fn }}
    # * b : `Tensor(U, OCL(U))` - RHS argument to {{ fn }}
    #
    # ## Examples
    #
    # ```
    # a = [1.5, 2.2, 3.2].to_tensor(OCL)
    # Num.{{ fn }}(a, a)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      \{% if U == Int32 %}
        singleton = Int32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == UInt32 %}
        singleton = UInt32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == Float32 %}
        singleton = Float32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == Float64 %}
        singleton = Float64{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% else %}
        \{% raise "Invalid Dtype" %}
      \{% end %}
    end

    # {{ fn.stringify.capitalize }} two `Tensor`s elementwise, storing
    # the result in the first Tensor
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to {{ fn }}, will be modified
    # inplace
    # * b : `Tensor(U, OCL(U))` - RHS argument to {{ fn }}
    #
    # ## Examples
    #
    # ```
    # a = [1.5, 2.2, 3.2].to_tensor(OCL)
    # Num.{{ fn }}!(a, a)
    # ```
    def {{ fn.id }}!(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Nil forall U
      \{% if U == Int32 %}
        singleton = Int32{{ fn.stringify.capitalize.id }}Inplace.instance
        singleton.call(a, b)
      \{% elsif U == UInt32 %}
        singleton = UInt32{{ fn.stringify.capitalize.id }}Inplace.instance
        singleton.call(a, b)
      \{% elsif U == Float32 %}
        singleton = Float32{{ fn.stringify.capitalize.id }}Inplace.instance
        singleton.call(a, b)
      \{% elsif U == Float64 %}
        singleton = Float64{{ fn.stringify.capitalize.id }}Inplace.instance
        singleton.call(a, b)
      \{% else %}
        \{% raise "Invalid Dtype" %}
      \{% end %}
    end
  end

  operator_op add, "+"
  operator_op subtract, "-"
  operator_op multiply, "*"
  operator_op divide, "/"

  macro relational_op(fn, operator)
    {% for dtype in [Int32, UInt32, Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::RelationalKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end
    {% end %}

    # Implements the comparison operator {{ operator }} between two `Tensor`s.
    # The returned result of OpenCL relational operators will always be
    # `Tensor(Int32, OCL(Int32))`.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to the {{ operator }} operator
    # * b : `Tensor(U, OCL(U))` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # b = [1, 8, 5.to_tensor(OCL)
    # Num.{{ fn }}(a, a)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      \{% if U == Int32 %}
        singleton = Int32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == UInt32 %}
        singleton = UInt32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == Float32 %}
        singleton = Float32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == Float64 %}
        singleton = Float64{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% else %}
        \{% raise "Invalid Dtype" %}
      \{% end %}
    end
  end

  relational_op greater, ">"
  relational_op greater_equal, ">="
  relational_op less, "<"
  relational_op less_equal, "<="
  relational_op equal, "=="
  relational_op not_equal, "!="

  macro bitwise_op(fn, operator)
    {% for dtype in [Int32, UInt32] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::RelationalKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end
    {% end %}

    # Implements the bitwise operator {{ operator }} between two `Tensor`s.
    # Only `Int32` and `UInt32` `Tensor`s are supported
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to the {{ operator }} operator
    # * b : `Tensor(U, OCL(U))` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # b = [1, 8, 5.to_tensor(OCL)
    # Num.{{ fn }}(a, a)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      \{% if U == Int32 %}
        singleton = Int32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% elsif U == UInt32 %}
        singleton = UInt32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a, b)
      \{% else %}
        \{% raise "Invalid Dtype" %}
      \{% end %}
    end
  end

  bitwise_op bitwise_and, "&"
  bitwise_op bitwise_or, "|"
  bitwise_op bitwise_xor, "^"
  bitwise_op left_shift, "<<"
  bitwise_op right_shift, ">>"

  macro builtin_op(fn)
    {% for dtype in [Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::BuiltinKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end
    {% end %}

    # Implements the OpenCL builtin function {{ fn.id }} for a single `Tensor`.
    # Only `Float32` and `Float64` `Tensor`s are supported.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` -`Tensor` on which to operate
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # Num.{{ fn.id }}(a)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      \{% if U == Float32 %}
        singleton = Float32{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a)
      \{% elsif U == Float64 %}
        singleton = Float64{{ fn.stringify.capitalize.id }}.instance
        singleton.call(a)
      \{% else %}
        \{% raise "Invalid dtype #{U} for OpenCL method {{ fn.id }}" %}
      \{% end %}
    end
  end

  builtin_op acos
  builtin_op acosh
  builtin_op acospi
  builtin_op asin
  builtin_op asinh
  builtin_op asinpi
  builtin_op atan
  builtin_op atanh
  builtin_op atanpi
  builtin_op cbrt
  builtin_op ceil
  builtin_op cos
  builtin_op cosh
  builtin_op cospi
  builtin_op erfc
  builtin_op erf
  builtin_op exp
  builtin_op exp2
  builtin_op exp10
  builtin_op expm1
  builtin_op fabs
  builtin_op floor
  builtin_op lgamma
  builtin_op log
  builtin_op log2
  builtin_op log10
  builtin_op log1p
  builtin_op logb
  builtin_op rint
  builtin_op round
  builtin_op rsqrt
  builtin_op sin
  builtin_op sinh
  builtin_op sinpi
  builtin_op sqrt
  builtin_op tan
  builtin_op tanh
  builtin_op tanpi
  builtin_op tgamma
  builtin_op trunc
end
