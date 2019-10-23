require "../core/vector"
require "../blas/level_one"

module Bottle::B
  extend self

  # Computes the total sum of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # sum(v) # => 10
  # ```
  def sum(a : Vector)
    a.reduce { |i, j| i + j }
  end

  # Computes the average of all vector values
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # mean(v) # => 2.5
  # ```
  def mean(a : Vector)
    a.sum / a.size
  end

  # Computes the standard deviation of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # std(v) # => 1.118
  # ```
  def std(a : Vector)
    avg = mean(a)
    r = power(a - avg, 2)
    Math.sqrt(r.sum / a.size)
  end

  # Computes the median value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # median(v) # => 2.5
  # ```
  def median(a : Vector)
    n = a.size
    sorted = a.sort
    if n % 2
      sorted[(n - 1) // 2]
    end
    m = (n - 1) / 2
    B.mean(sorted[[m.floor.to_i32, m.ceil.to_i32]])
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # max(v) # => 4
  # ```
  def max(a : Vector)
    max_helper(a)[0]
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # argmax(v) # => 3
  # ```
  def argmax(a : Vector)
    max_helper(a)[1]
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # max_helper(v) # => {true, 4, 3}
  # ```
  private def max_helper(v : Vector(U)) forall U
    max = uninitialized U
    index = uninitialized Int32

    v.each_with_index do |elem, i|
      if i == 0 || elem > max
        max = elem
        index = i
      end
    end
    {max, index}
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # min(v) # => 1
  # ```
  def min(a : Vector(U)) forall U
    min_helper(a)[0]
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # argmin(v) # => 0
  # ```
  def argmin(a : Vector(U)) forall U
    min_helper(a)[1]
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # min_helper(v) # => {true, 4, 3}
  # ```
  private def min_helper(v : Vector(U)) forall U
    min = uninitialized U
    index = uninitialized Int32

    v.each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        index = i
      end
    end
    {min, index}
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp(v : Vector)
    min, max, _, _ = ptp_helper(v)
    max - min
  end

  # Internal method to find the minimum and maximum values,
  # as well as the respective indexes for a flask.
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # ptp_helper(v) # => {true, 1, 4, 0, 3}
  # ```
  private def ptp_helper(v : Vector(U)) forall U
    min = uninitialized U
    max = uninitialized U
    imin = uninitialized Int32
    imax = uninitialized Int32

    v.each_with_index do |elem, i|
      if i == 0 || elem < min
        min = elem
        imin = i
      end
      if i == 0 || elem > max
        max = elem
        imax = i
      end
    end
    {min, max, imin, imax}
  end
end
