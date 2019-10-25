require "../core/tensor"
require "../core/matrix"

class Bottle::UFunc::Accumulate(T)
  getter data : Tensor(T)
  getter size : Int32
  getter inplace : Bool

  def initialize(data : Tensor(T), @inplace = false)
    @size = data.size
    if !inplace
      @data = data.clone
    else
      @data = data
    end
  end

  def to_s(io)
    io << "<ufunc> accumulate"
  end

  # Accumulates sum of a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.sum # => [1, 3, 6, 10]
  # ```
  def add
    (1...size).each do |i|
      data[i] += data[i - 1]
    end
    return data unless inplace
  end

  # Accumulates difference of a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.subtract # => [1, -1, -4, -8]
  # ```
  def subtract
    (1...size).each do |i|
      data[i] -= data[i - 1]
    end
    return data unless inplace
  end

  # Accumulates product of a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.multiply # => [1, 2, 6, 12]
  # ```
  def multiply
    (1...size).each do |i|
      data[i] *= data[i - 1]
    end
    return data unless inplace
  end

  # Accumulates division of a Flask.
  # The input flask is cast to a double
  # type first.
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.divide # => [1.0, 0.5, 6, 2.4]
  # ```
  def divide
    (1...size).each do |i|
      data[i] /= data[i - 1]
    end
    return data unless inplace
  end

  # Accumulates minimum value of a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.minimum # => [1, 1, 1, 1]
  # ```
  def minimum
    (1...size).each do |i|
      data[i] = Math.min(data[i], data[i - 1])
    end
    return data unless inplace
  end

  # Accumulates maximum value of a Flask
  #
  # ```
  # v = Flask.new [1, 2, 3, 4]
  # v.accumulate.maximum # => [1, 2, 3, 4]
  # ```
  def maximum
    (1...size).each do |i|
      data[i] = Math.max(data[i], data[i - 1])
    end
    return data unless inplace
  end
end

class Bottle::UFunc::Accumulate2D(T)
  getter data : Matrix(T)
  getter axis : Int32?
  getter inplace : Bool

  def initialize(data : Matrix(T), @axis, @inplace = false)
    @data = inplace ? data : data.clone
  end

  def apply_along_axis(&block : Tensor(T) -> Nil)
    if axis == 0
      data.each_col do |e|
        yield e
      end
    else
      data.each_row do |e|
        yield e
      end
    end
    return data unless inplace
  end

  # Accumulates the sum of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).add # =>
  # [[1.0, 3.0, 6.0]
  #  [4.0, 9.0, 15.0]
  #  [7.0, 15.0, 24.0]]
  # ```
  def add
    if axis.nil?
      data.ravel.accumulate.add
    end
    apply_along_axis &.accumulate(true).add
  end

  # Accumulates the subtraction of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).subtract # =>
  # [[1.0, 1.0, 2.0]
  #  [4.0, 1.0, 5.0]
  #  [7.0, 1.0, 8.0]]
  # ```
  def subtract
    if axis.nil?
      data.ravel.accumulate.subtract
    end
    apply_along_axis &.accumulate(true).subtract
  end

  # Accumulates the multiplication of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).multiply # =>
  # [[1.0, 2.0, 6.0]
  #  [4.0, 20.0, 120.0]
  #  [7.0, 56.0, 504.0]]
  # ```
  def multiply
    if axis.nil?
      data.ravel.accumulate.multiply
    end
    apply_along_axis &.accumulate(true).multiply
  end

  # Accumulates the division of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).divide # =>
  # [[1.0, 2.0, 1.5]
  #  [4.0, 1.25, 4.8]
  #  [7.0, 1.143, 7.875]]
  # ```
  def divide
    if axis.nil?
      data.ravel.accumulate.divide
    end
    apply_along_axis &.accumulate(true).divide
  end

  # Accumulates the minimum of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).minimum # =>
  # [[1.0, 1.0, 1.0]
  #  [4.0, 4.0, 4.0]
  #  [7.0, 7.0, 7.0]]
  # ```
  def minimum
    if axis.nil?
      data.ravel.accumulate.minimum
    end
    apply_along_axis &.accumulate(true).minimum
  end

  # Accumulates the maximum of a Jug along an axis or
  # a flattened view of the Jug
  #
  # ```crystal
  # j = Jug.new [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  # j.accumulate(axis=1).minimum # =>
  # [[1.0, 2.0, 3.0]
  #  [4.0, 5.0, 6.0]
  #  [7.0, 8.0, 9.0]]
  # ```
  def maximum
    if axis.nil?
      data.ravel.accumulate.maximum
    end
    apply_along_axis &.accumulate(true).maximum
  end
end
