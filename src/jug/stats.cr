require "../libs/dtype"
require "./*"
require "../blas/*"

class Jug(T)
  # Computes the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    ravel.max
  end

  def max(axis : Indexer)
    reduce(axis, &.max)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    ravel.min
  end

  def min(axis : Indexer)
    reduce(axis, &.max)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    {min, max}
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    max - min
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    ravel.argmax
  end

  def argmax(axis : Indexer)
    reduce(axis, &.argmax)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    ravel.argmin
  end

  def argmin(axis : Indexer)
    reduce(axis, &.argmin)
  end

  # Computes the cumulative sum of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum # => [1, 3, 6, 10]
  # ```
  def cumsum
    self.clone.ravel.cumsum
  end

  def cumsum(axis : Indexer)
    ret = self.clone
    ret.accumulate(axis, &.cumsum!)
    ret
  end

  # Computes the cumulative product of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod # => [1, 2, 6, 24]
  # ```
  def cumprod
    self.clone.ravel.cumprod
  end

  def cumprod(axis : Indexer)
    ret = self.clone
    accumulate(axis, &.cumprod!)
    ret
  end
end
