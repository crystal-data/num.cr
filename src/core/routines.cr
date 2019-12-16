require "../tensor/tensor"

module Num::Routines
  def average(a)
    mean(a)
  end

  def average(a, axis : Int32)
    mean(a, axis: axis)
  end
end
