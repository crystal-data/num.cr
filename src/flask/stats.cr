require "../libs/dtype"
require "./*"
require "../blas/*"

class Flask(T)
  # Computes the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max
    data.max
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    data.min
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    data.minmax
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    mn, mx = data.minmax
    return mx - mn
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    index = 0
    mx = data[0]
    data.each_with_index { |e, i|  index = i unless e < mx }
    return index
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    index = 0
    mn = data[0]
    data.each_with_index { |e, i|  index = i unless e > mn }
    return index
  end

  # Computes the cumulative sum of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum # => [1, 3, 6, 10]
  # ```
  def cumsum
    ret = self.clone
    (1...ret.size).each { |index| ret[index] += ret[index -1] }
    return ret
  end

  # Computes the cumulative product of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod # => [1, 2, 6, 24]
  # ```
  def cumprod
    ret = self.clone
    (1...ret.size).each { |index| ret[index] *= ret[index -1] }
    return ret
  end
end
