require "./*"
require "../gsl/*"
require "../libs/gsl"

class Vector(T, D)
  # Elementwise addition of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 + v2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Vector(T, D))
    LL.add(self.copy, other)
  end

  # Elementwise addition of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 + v2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : BNum)
    LL.add(self.copy, other)
  end

  # Elementwise subtraction of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 - v2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Vector(T, D))
    LL.sub(self.copy, other)
  end

  # Elementwise subtraction of a vector with a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 - v2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : BNum)
    LL.sub(self.copy, other)
  end

  # Elementwise multiplication of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 * v2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Vector(T, D))
    LL.mul(self.copy, other)
  end

  # Elementwise multiplication of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 + v2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : BNum)
    LL.mul(self.copy, other)
  end

  # Elementwise division of a vector to another equally sized vector
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = Vector.new [2.0, 4.0, 6.0]
  # v1 / v2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Vector(T, D))
    LL.div(self.copy, other)
  end

  # Elementwise division of a vector to a scalar
  #
  # ```
  # v1 = Vector.new [1.0, 2.0, 3.0]
  # v2 = 2
  # v1 / v2 # => [0.5, 1, 1.5]
  # ```
  def /(other : BNum)
    LL.div(self.copy, other)
  end
end
