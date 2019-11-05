require "./ndtensor"

# A module primarily responsible for `Tensor`
# and `Matrix` trigonometric routines.
#
# This module should be namespaced as part of the
# external API to provide user facing methods
# for creation.
module Bottle::Internal::Trigonometric
  extend self

  macro trig(names)
    {% for name in names %}
      # Calculates the {{name}} of a `Tensor`
      #
      # ```
      # t1 = Tensor.new [1, 2, 3]
      #
      # B.{{name}}(t1)
      # ```
      def {{name}}(x1 : Tensor)
        iter = x1.unsafe_iter
        Tensor.new(x1.shape) do |_|
          Math.{{name}}(iter.next.value)
        end
      end
    {% end %}
  end

  trig [
    acos, acosh, asin, asinh, atan, atanh, besselj0, besselj1, bessely0,
    bessely1, cbrt, cos, cosh, erf, erfc, exp, exp2, expm1, frexp, gamma,
    ilogb, lgamma, log, log10, log1p, log2, logb, sin, sinh, sqrt, tan,
    tanh,
  ]

  macro trig2d(names)
    {% for name in names %}
      # Computes the {{name}} of two Tensors elementwise
      #
      # ```
      # t1 = Tensor.new [1, 2, 3]
      # t2 = Tensor.new [4, 5, 6]
      #
      # B.{{name}}(t1, t2)
      # ```
      def {{name}}(x1 : Tensor, x2 : Tensor)
        if x1.shape != x2.shape
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end

        i1 = x1.unsafe_iter
        i2 = x2.unsafe_iter

        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.shape) do |_|
          Math.{{name}}(i1.next.value, i2.next.value)
        end
      end

      # Computes the {{name}} of a `Tensor` with a scalar
      # elementwise
      #
      # ```
      # t1 = Tensor.new [1, 2, 3]
      # t2 = 5
      #
      # B.{{name}}(t1, t2)
      # ```
      def {{name}}(x1 : Tensor, x2 : Number)
        iter = x1.unsafe_iter
        Tensor.new(x1.shape) do |_|
          Math.{{name}}(iter.next.value, x2)
        end
      end

      # {{name}}s a scalar with a tensor elementwise.
      #
      # ```
      # x = 5
      # t = Tensor.new [1, 2, 3]
      #
      # B.{{name}}(x, t)
      # ```
      def {{name}}(x1 : Number, x2 : Tensor, where : Tensor? = nil)
        iter = x2.unsafe_iter
        Tensor.new(x2.shape) do |_|
          Math.{{name}}(x2, iter.next.value)
        end
      end

      # Returns the universal {{name}} function. Used to
      # apply outer operations, reductions, and accumulations
      # to tensors
      #
      # B.{{name}} # => <ufunc> {{name}}
      def {{name}}
        UFunc_{{name}}.new
      end

      # :nodoc:
      struct UFunc_{{name}}

        # A basic string representation of a
        # universal function.
        #
        # TODO: Add the same string representation
        # to the functions the struct contains
        def to_s(io)
          io << "<ufunc> {{ name }}"
        end

        # Applies an outer operations between two `Tensor`s.
        # Returns an MxN matrix where M is the size of *x1*,
        # and N is the size of *x2*
        #
        # ```
        # t = Tensor.new [1, 2]
        #
        # puts B.hypot.outer(t, t)
        #
        # # Matrix[[  2  3]
        # #        [  3  4]]
        # ```
        def outer(x1 : Tensor, x2 : Tensor)
          outer = x1.unsafe_iter
          inner = x2.unsafe_iter
          c1 = uninitialized U
          c2 = uninitialized V
          Tensor.new(x1.shape + x2.shape) do |i|
            d = i % x2.size
            if d == 0
              c1 = outer.next.value
              inner = x2.unsafe_iter
            end
            Math.{{name}}(c1, inner.next.value)
          end
        end
      end
    {% end %}
  end

  trig2d [atan2, besselj, bessely, copysign, hypot, ldexp]

  # Convert angles from radians to degrees.
  #
  # ```
  # t = Tensor.new [0, 1, 2, 3] * (Math::PI / 6)
  #
  # degrees(t) # => Tensor[      0.0     30.0     60.0     90.0]
  # ```
  def degrees(x1 : Tensor)
    iter = x1.unsafe_iter
    Tensor.new(x1.shape) do |_|
      iter.next * (180/Math::PI)
    end
  end

  # Convert angles from degrees to radians
  #
  # ```
  # t = Tensor.new [30, 60, 90, 120]
  #
  # radians(t) # => Tensor[     0.524     1.047     1.571     2.094]
  # ```
  def radians(x1 : Tensor)
    iter = x1.unsafe_iter
    Tensor.new(x1.size) do |_|
      iter.next * (Math::PI/180)
    end
  end
end
