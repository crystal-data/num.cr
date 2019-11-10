require "./base"
require "./flags"

module Bottle::Internal::BaseHelpers
  extend self
  macro contiguity_checks(order)
    def is_obj_{{order}}(obj : BaseArray)
      if obj.ndims == 0
        return true
      end

      if obj.ndims == 1
        return obj.shape[0] == 0 || obj.strides[0] == 1
      end

      sd = 1

      {% if order == :fortran %}
        obj.ndims.times do |d|
          dim = obj.shape[d]
          if obj.strides[d] != sd
            return false
          end
          sd *= dim
        end
      {% elsif order == :contiguous %}
        (obj.ndims - 1).step(to: 0, by: -1) do |d|
          dim = obj.shape[d]
          if obj.strides[i] != sd
            return false
          end
        end
      {% end %}
      true
    end
  end
  contiguity_checks(fortran)
  contiguity_checks(contiguous)
end
