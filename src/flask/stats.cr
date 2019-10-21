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
    _, max, _ = max_internal
    max
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax
    _, _, index = max_internal
    index
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
  # ```
  private def max_internal
    max = uninitialized T
    index = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem > max
        max = elem
        index = i
      end
      found = true
    end

    {found, max, index}
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min
    _, min, _ = min_internal
    min
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin
    _, _, index = min_internal
    index
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
  # ```
  private def min_internal
    min = uninitialized T
    index = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        index = i
      end
      found = true
    end

    {found, min, index}
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def ptpv
    _, min, max, _, _ = ptp_internal
    return {min, max}
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    min, max = ptpv
    return max - min
  end

  # Internal method to find the minimum and maximum values,
  # as well as the respective indexes for a flask.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp_internal # => {true, 1, 4, 0, 3}
  # ```
  private def ptp_internal
    min = uninitialized T
    max = uninitialized T
    imin = uninitialized Int32
    imax = uninitialized Int32
    found = false

    each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        imin = i
      end
      if i == 0 || elem > max
        max = elem
        imax = i
      end
      found = true
    end
    {found, min, max, imin, imax}
  end

  # Computes the cumulative sum of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum # => [1, 3, 6, 10]
  # ```
  def cumsum
    ret = self.clone
    ret.cumsum!
    ret
  end

  # Computes the cumulative sum of a vector in place.
  # Primarily used for reductions along an axis in
  # a Jug.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumsum!
  # v # => [1, 3, 6, 10]
  # ```
  def cumsum!
    (1...size).each do |i|
      self[i] += self[i - 1]
    end
  end

  # Computes the cumulative product of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod # => [1, 2, 6, 24]
  # ```
  def cumprod
    ret = self.clone
    ret.cumprod!
    ret
  end

  # Computes the cumulative product of a vector in place.
  # Primarily used for reductions along an axis in
  # a Jug.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.cumprod!
  # v # => [1, 2, 6, 24]
  # ```
  def cumprod!
    (1...size).each do |i|
      self[i] *= self[i - 1]
    end
  end
end
