require "../core/vector"
require "../blas/level_one"

module Bottle::B
  extend self

  # Computes the sum of each value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.sum # => 10
  # ```
  def sum(a : Vector)
    a.reduce { |i, j| i + j }
  end

  def mean(a : Vector)
    a.sum / a.size
  end

  def std(a : Vector)
    avg = mean(a)
    r = power(a - avg, 2)
    Math.sqrt(r.sum / a.size)
  end

  # Computes the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def max(a : Vector)
    max_helper(a)[0]
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmax # => 3
  # ```
  def argmax(a : Vector)
    max_helper(a)[1]
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
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
  # v = Flask.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def min(a : Vector(U)) forall U
    min_helper(a)[0]
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.argmin # => 0
  # ```
  def argmin(a : Vector(U)) forall U
    min_helper(a)[1]
  end

  # Internal method to find the maximum value and the index
  # of the maximum value for a Vector
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.max_internal # => {true, 4, 3}
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
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp(v : Vector)
    min, max, _, _ = ptp_internal(v)
    max - min
  end

  # Internal method to find the minimum and maximum values,
  # as well as the respective indexes for a flask.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.ptp_internal # => {true, 1, 4, 0, 3}
  # ```
  private def ptp_internal(v : Vector(U)) forall U
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
