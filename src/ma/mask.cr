require "../core/tensor"
require "../core/matrix"

module Bottle
  macro mask_helper(operators)
    module B
      extend self

      def greater(x : Number, y : Number)
        x > y
      end

      def greater_equal(x : Number, y : Number)
        x >= y
      end

      def less(x : Number, y : Number)
        x < y
      end

      def less_equal(x : Number, y : Number)
        x <= y
      end

      def equal(x : Number, y : Number)
        x == y
      end

      {% for op in operators %}
        def {{op.id}}(a : Vector, b : Vector)
          Vector.new(a.size) { |i| {{op.id}}(a[i], b[i]) }
        end

        def {{op.id}}(a : Vector, x : Number)
          Vector.new(a.size) { |i| {{op.id}}(a[i], x) }
        end

        def {{op.id}}(a : Matrix, b : Matrix)
          Matrix.new(a.nrows, a.ncols) { |i, j| {{op.id}}(a[i, j], b[i, j]) }
        end

        def {{op.id}}(a : Matrix, x : Number)
          Matrix.new(a.nrows, a.ncols) { |i, j| {{op.id}}(a[i, j], x) }
        end
      {% end %}
    end
  end

  mask_helper [greater, greater_equal, less, less_equal, equal]
end
