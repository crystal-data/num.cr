require "../core/flask"
require "../core/jug"

module Bottle
  macro mask_helper(operators)
    module MA
      extend self

      def gt(x : Number, y : Number)
        x > y
      end

      def ge(x : Number, y : Number)
        x >= y
      end

      def lt(x : Number, y : Number)
        x < y
      end

      def le(x : Number, y : Number)
        x <= y
      end

      def eq(x : Number, y : Number)
        x == y
      end

      {% for op in operators %}

        def {{op.id}}(a : Flask, b : Flask)
          Flask(Bool).new(a.size) { |i| {{op.id}}(a[i], b[i]) }
        end

        def {{op.id}}(a : Flask, x : Number)
          Flask(Bool).new(a.size) { |i| {{op.id}}(a[i], x) }
        end

        def {{op.id}}(a : Jug, b : Jug)
          Jug(Bool).new(a.nrows, a.ncols) { |i, j| {{op.id}}(a[i, j], b[i, j]) }
        end

        def {{op.id}}(a : Jug, x : Number)
          Jug(Bool).new(a.nrows, a.ncols) { |i, j| {{op.id}}(a[i, j], x) }
        end

      {% end %}

    end
  end

  mask_helper [gt, ge, lt, le, eq]
end
