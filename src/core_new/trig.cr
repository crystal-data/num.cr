require "../core/tensor"
require "../core/matrix"

# A module primarily responsible for `Tensor`
# and `Matrix` trigonometric routines.
#
# This module should be namespaced as part of the
# external API to provide user facing methods
# for creation.
module Bottle::Internal
  macro trig(names)
    module Trigonometric
      extend self

      {% for name in names %}
        # Calculates the {{name}} of a `Tensor`
        #
        # t1 = Tensor.new [1, 2, 3]
        #
        # B.{{name}}(t1)
        def {{name}}(x1 : Tensor, where : Tensor? = nil)

          # TODO: Implement masking to use the *where* parameter
          Tensor.new(x1.size) do |i|
            Math.{{name}}(x1[i])
          end
        end

        # Calculates the {{name}} of a `Tensor`, storing
        # the result in *dest*
        #
        # t1 = Tensor.new [1, 2, 3]
        #
        # B.{{name}}(t1, dest: t2)
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
  end

  trig [acos, acosh, asin, asinh, atan, atanh, besselj0, besselj1, bessely0,
        bessely1, cbrt, cos, cosh, erf, erfc, exp, exp2, expm1, frexp, gamma,
        ilogb, lgamma, log, log10, log1p, log2, logb, sin, sinh, sqrt, tan,
        tanh,
  ]
end

include Bottle::Internal::Trigonometric
include Bottle

t = Tensor.new [1, 2, 3, 4, 5]
puts log10(t)
