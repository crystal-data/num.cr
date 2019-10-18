require "./*"
require "../gsl/*"
require "../libs/gsl"

class Vector(T, D)
  # Computes the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    LL.max(self)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    LL.min(self)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    LL.ptpv(self)
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    LL.ptp(self)
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    LL.argmax(self)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    LL.argmin(self)
  end

  # Computes the indexes of the minimum and maximum values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.argminmax # => {0, 3}
  # ```
  def argminmax
    LL.argminmax(self)
  end

  # Computes the cumulative sum of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.cumsum # => [1, 3, 6, 10]
  # ```
  def cumsum
    LL.cumsum(self.copy)
  end

  # Computes the cumulative product of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.cumprod # => [1, 2, 6, 24]
  # ```
  def cumprod
    LL.cumprod(self.copy)
  end
end
