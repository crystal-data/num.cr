require "../core/ndtensor"
require "./fixed_dimension"
require "../util/exceptions"

module Bottle::Internal::LinAlg
  macro matrix_reduction_retain_size(operation)
    def {{operation}}(t : Tensor)
      if t.ndims < 2
        raise Exceptions::ShapeError.new("Tensor must be at least 2D")
      end

      if t.ndims == 2
        return {{operation}}_helper(t)
      end

      t = t.dup
      t.matrix_iter.each do |m|
        m[...] = {{operation}}_helper(m)
      end
      t
    end
  end

  matrix_reduction_retain_size(inv)
end
