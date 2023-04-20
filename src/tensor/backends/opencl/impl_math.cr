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
  # :nodoc:
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

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}TensorScalar < Num::ArithmeticTensorScalarKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}TensorScalarInplace < Num::ArithmeticTensorScalarInplaceKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}ScalarTensor < Num::ArithmeticScalarTensorKernel({{ dtype }})
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
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }},
        [Int32, UInt32, Float32, Float64],
        a, b
      )
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
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}Inplace,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end

    # {{ fn.stringify.capitalize.id }} a `Tensor` and a `Number` elementwise
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to {{ fn }}
    # * b : `U` - RHS argument to {{ fn }}
    #
    # ## Examples
    #
    # ```
    # a = [1.5, 2.2, 3.2].to_tensor(OCL)
    # Num.{{ fn }}(a, 3.5)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : U) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}TensorScalar,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end

    # {{ fn.stringify.capitalize.id }} a `Tensor` and a `Number` elementwise,
    # modifying the `Tensor` inplace.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to {{ fn }}
    # * b : `U` - RHS argument to {{ fn }}
    #
    # ## Examples
    #
    # ```
    # a = [1.5, 2.2, 3.2].to_tensor(OCL)
    # Num.{{ fn }}(a, 3.5)
    # ```
    def {{ fn.id }}!(a : Tensor(U, OCL(U)), b : U) : Nil forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}TensorScalarInplace,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end

    # {{ fn.stringify.capitalize.id }} a `Number` and a `Tensor` elementwise
    #
    # ## Arguments
    #
    # * a : `U` - LHS argument to {{ fn }}
    # * b : `Tensor(U, OCL(U))` - RHS argument to {{ fn }}
    #
    # ## Examples
    #
    # ```
    # a = [1.5, 2.2, 3.2].to_tensor(OCL)
    # Num.{{ fn }}(a, 3.5)
    # ```
    def {{ fn.id }}(a : U, b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}ScalarTensor,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end
  end

  operator_op add, "+"
  operator_op subtract, "-"
  operator_op multiply, "*"
  operator_op divide, "/"

  # :nodoc:
  macro relational_op(fn, operator)
    {% for dtype in [Int32, UInt32, Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::RelationalKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}TensorScalar < Num::RelationalTensorScalarKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}ScalarTensor < Num::RelationalScalarTensorKernel({{ dtype }})
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
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Tensor(Int32, OCL(Int32)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }},
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end

    # Implements the comparison operator {{ operator }} between a `Tensor`
    # and a `Number`.
    # `
    # The returned result of OpenCL relational operators will always be
    # `Tensor(Int32, OCL(Int32))`.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to the {{ operator }} operator
    # * b : `U` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # Num.{{ fn }}(a, 3)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : U) : Tensor(Int32, OCL(Int32)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}TensorScalar,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end

    # Implements the comparison operator {{ operator }} between a `Number`
    # and a `Tensor`.
    # `
    # The returned result of OpenCL relational operators will always be
    # `Tensor(Int32, OCL(Int32))`.
    #
    # ## Arguments
    #
    # * a : `U` - LHS argument to the {{ operator }} operator
    # * b : `Tensor(U, OCL(U))` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # Num.{{ fn }}(3, a)
    # ```
    def {{ fn.id }}(a : U, b : Tensor(U, OCL(U))) : Tensor(Int32, OCL(Int32)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}ScalarTensor,
        [Int32, UInt32, Float32, Float64],
        a, b
      )
    end
  end

  relational_op greater, ">"
  relational_op greater_equal, ">="
  relational_op less, "<"
  relational_op less_equal, "<="
  relational_op equal, "=="
  relational_op not_equal, "!="

  # :nodoc:
  macro bitwise_op(fn, operator)
    {% for dtype in [Int32, UInt32] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::ArithmeticKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}TensorScalar < Num::ArithmeticTensorScalarKernel({{ dtype }})
        @@operator = "{{ operator.id }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}ScalarTensor < Num::ArithmeticScalarTensorKernel({{ dtype }})
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
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }},
        [Int32, UInt32],
        a, b
      )
    end

    # Implements the bitwise operator {{ operator }} between a `Tensor` and
    # a `Number`
    #
    # Only `Int32` and `UInt32` `Tensor`s are supported
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS argument to the {{ operator }} operator
    # * b : `U` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # Num.{{ fn }}(a, 3)
    # ```
    def {{ fn.id }}(a : Tensor(U, OCL(U)), b : U) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}TensorScalar,
        [Int32, UInt32],
        a, b
      )
    end

    # Implements the bitwise operator {{ operator }} between a `Tensor` and
    # a `Number`
    #
    # Only `Int32` and `UInt32` `Tensor`s are supported
    #
    # ## Arguments
    #
    # * a : `U` - LHS argument to the {{ operator }} operator
    # * b : `Tensor(U, OCL(U))` - RHS argument to the {{ operator }} operator
    #
    # ## Examples
    #
    # ```
    # a = [12, 3, 5].to_tensor(OCL)
    # Num.{{ fn }}(3, a)
    # ```
    def {{ fn.id }}(a : U, b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}ScalarTensor,
        [Int32, UInt32],
        a, b
      )
    end
  end

  bitwise_op bitwise_and, "&"
  bitwise_op bitwise_or, "|"
  bitwise_op bitwise_xor, "^"
  bitwise_op left_shift, "<<"
  bitwise_op right_shift, ">>"

  # :nodoc:
  macro builtin_op(fn)
    {% for dtype in [Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::BuiltinKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}Inplace < Num::BuiltinInplaceKernel({{ dtype }})
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
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }},
        [Float32, Float64],
        a
      )
    end

    # Implements the OpenCL builtin function {{ fn.id }} for a single `Tensor`.
    # Only `Float32` and `Float64` `Tensor`s are supported.  This method
    # mutates the original `Tensor`, modifying it in place.
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` -`Tensor` on which to operate
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # Num.{{ fn.id }}!(a)
    # ```
    def {{ fn.id }}!(a : Tensor(U, OCL(U))) : Nil forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}Inplace,
        [Float32, Float64],
        a
      )
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

  # :nodoc:
  macro builtin_two_op(fn, name)
    {% for dtype in [Float32, Float64] %}
      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }} < Num::BuiltinTwoKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}Inplace < Num::BuiltinTwoInplaceKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}TensorScalar < Num::BuiltinTwoTensorScalarKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end

      # :nodoc:
      class {{ dtype }}{{ fn.stringify.capitalize.id }}ScalarTensor < Num::BuiltinTwoScalarTensorKernel({{ dtype }})
        @@fn = "{{ fn }}"
        @@name = "{{ fn }}Kernel"
      end
    {% end %}

    # Implements the OpenCL builtin function {{ fn.id }} between two `Tensor`s
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS `Tensor` for the operation
    # * b : `Tensor(U, OCL(U))` - RHS `Tensor` for the operation
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # b = [0.2, 0.5, 0.1].to_tensor(OCL)
    # Num.{{ fn.id }}(a, b)
    # ```
    def {{ name.id }}(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }},
        [Float32, Float64],
        a, b
      )
    end

    # Implements the OpenCL builtin function {{ fn.id }} between two `Tensor`s,
    # mutating the first `Tensor`, modifying it to store the result of the
    # operation
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS `Tensor` for the operation
    # * b : `Tensor(U, OCL(U))` - RHS `Tensor` for the operation
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # b = [0.2, 0.5, 0.1].to_tensor(OCL)
    # Num.{{ fn.id }}!(a, b)
    # ```
    def {{ name.id }}!(a : Tensor(U, OCL(U)), b : Tensor(U, OCL(U))) : Nil forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}Inplace,
        [Float32, Float64],
        a, b
      )
    end

    # Implements the OpenCL builtin function {{ fn.id }} between two a `Tensor`
    # and a `Number`
    #
    # ## Arguments
    #
    # * a : `Tensor(U, OCL(U))` - LHS `Tensor` for the operation
    # * b : `U` - RHS `Number` for the operation
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # Num.{{ fn.id }}(a, 3_f64)
    # ```
    def {{ name.id }}(a : Tensor(U, OCL(U)), b : U) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}TensorScalar,
        [Float32, Float64],
        a, b
      )
    end

    # Implements the OpenCL builtin function {{ fn.id }} between two a `Tensor`
    # and a `Number`
    #
    # ## Arguments
    #
    # * a : `U` - LHS `Number` for the operation
    # * b : `Tensor(U, OCL(U))` - RHS `Tensor` for the operation
    #
    # ## Examples
    #
    # ```
    # a = [0.45, 0.3, 2.4].to_tensor(OCL)
    # Num.{{ fn.id }}(3_f64, a)
    # ```
    def {{ name.id }}(a : U, b : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
      call_opencl_kernel(
        U,
        {{ fn.stringify.capitalize.id }}ScalarTensor,
        [Float32, Float64],
        a, b
      )
    end
  end

  builtin_two_op pow, power
  builtin_two_op atan2, atan2
  builtin_two_op fmax, max
  builtin_two_op fmin, min

  # Implements the negation operator on a `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, OCL(U))` - `Tensor` to negate
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # Num.negate(a) # => [-1, -2, -3]
  # ```
  def negate(a : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
    U.new(0) - a
  end
end
