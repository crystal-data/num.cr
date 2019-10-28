require "./tensor"
require "./matrix"

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
      def {{name}}(x1 : Tensor, where : Tensor? = nil)

        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.size) do |i|
          Math.{{name}}(x1[i])
        end
      end

      # Calculates the {{name}} of a `Tensor`, storing
      # the result in *dest*
      #
      # ```
      # t1 = Tensor.new [1, 2, 3]
      #
      # B.{{name}}(t1, dest: t2)
      # ```
      def {{name}}(x1 : Tensor, dest : Tensor, where : Tensor? = nil)
        if x1.size != dest.size
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end
        # TODO: Implement masking to use the *where* parameter
        x1.size.times do |i|
          dest[i] = Math.{{name}}(x1[i])
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
      def {{name}}(x1 : Tensor, x2 : Tensor, where : Tensor? = nil)
        if x1.size != x2.size
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end

        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.size) do |i|
          Math.{{name}}(x1[i], x2[i])
        end
      end

      # Computes the {{name}} of two Tensors elementwise, storing
      # the result in *dest*
      #
      # ```
      # t1 = Tensor.new [1, 2, 3]
      # t2 = Tensor.empty(t1.size)
      #
      # B.{{name}}(t1, t1, dest: t2)
      # ```
      def {{name}}(x1 : Tensor, x2 : Tensor, dest : Tensor, where : Tensor? = nil)
        if x1.size != x2.size
          raise "Shapes {#{x1.size}} and {#{x2.size} are not aligned"
        end

        # TODO: Implement masking to use the *where* parameter
        x1.size.times do |i|
          dest[i] = Math.{{name}}(x1[i], x2[i])
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
      def {{name}}(x1 : Tensor, x2 : Number, where : Tensor? = nil)
        # TODO: Implement masking to use the *where* parameter
        Tensor.new(x1.size) do |i|
          Math.{{name}}(x1[i], x2)
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
        {{name}}(x2, x1, where)
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
          Matrix.new(x1.size, x2.size) do |i, j|
            Math.{{name}}(x1[i], x2[j])
          end
        end

        # Applies an accumulation function along
        # a `Tensor`.  Returns a copy of the `Tensor`
        #
        # ```
        # t = Tensor.new [1, 2, 3, 4, 5]
        #
        # t.add.accumulate # => [1, 3, 6, 10, 15]
        # ```
        def accumulate(x1 : Tensor)
          ret = x1.clone
          (1...x1.size).each do |i|
            ret[i] = Math.{{name}}(ret[i], ret[i - 1])
          end
          ret
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
    Tensor.new(x1.size) do |i|
      x1[i] * (180/Math::PI)
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
    Tensor.new(x1.size) do |i|
      x1[i] * (Math::PI/180)
    end
  end
end
