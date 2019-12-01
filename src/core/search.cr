require "./macros"
require "./common"

module Bottle::Internal
  def where(condition : BaseArray(Bool), x : BaseArray(U), y : BaseArray(U)) forall U
    broadcast_rhs condition, x
    broadcast_rhs condition, y
    ret = Tensor(U).new(condition.shape)
    ret.flat_iter.zip(condition.flat_iter, x.flat_iter, y.flat_iter) do |v, i, j, k|
      v.value = i.value ? j.value : k.value
    end
    ret
  end

  def where(condition : BaseArray(Bool), x : BaseArray)
    broadcast_rhs condition, x
    ret = Tensor(Float64).new(condition.shape)
    ret.flat_iter.zip(condition.flat_iter, x.flat_iter) do |v, i, j|
      v.value = i.value ? Float64.new(j.value) : Float64::NAN
    end
    ret
  end
end
