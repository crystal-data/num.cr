require "./*"
require "../libs/dtype"
require "../flask/*"
require "../strides/offsets"
require "../indexing/base"

class Jug(T)
  # Elementwise addition of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 + j1 # => [[2, 4], [6, 8]]
  # ```
  def +(other : Jug(T))
    LL.add(self.clone, other)
  end

  # Elementwise addition of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 + 2 # => [[3, 4], [5, 6]]
  # ```
  def +(other : T)
    LL.add(self.clone, other)
  end

  # Elementwise subtraction of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 - j1 # => [[0, 0], [0, 0]]
  # ```
  def -(other : Jug(T))
    LL.sub(self.clone, other)
  end

  # Elementwise subtraction of a Flask with a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 - 2 # => [[-1, 0], [1, 2]]
  # ```
  def -(other : T)
    LL.sub(self.clone, other)
  end

  # Elementwise multiplication of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 * j1 # => [[1, 4], [9, 16]]
  # ```
  def *(other : Jug(T))
    LL.mul(self.clone, other)
  end

  # Elementwise multiplication of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 * 2 # => [[2, 4], [6, 8]]
  # ```
  def *(other : T)
    LL.mul(self.clone, other)
  end

  # Elementwise division of a Flask to another equally sized Flask
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 / j1 # => [[1, 1], [1, 1]]
  # ```
  def /(other : Jug(T))
    LL.div(self.clone, other)
  end

  # Elementwise division of a Flask to a scalar
  #
  # ```
  # j1 = Jug.new [[1, 2], [3, 4]]
  # j1 / 2 # => [[0.5, 1.0], [1.5, 2.0]]
  # ```
  def /(other : T)
    LL.div(self.clone, other)
  end

  def >(other : Jug)
    LL.gt(self, other)
  end

  def >(other : Number)
    LL.gt(self, other)
  end

  def >=(other : Jug)
    LL.ge(self, other)
  end

  def >=(other : Number)
    LL.ge(self, other)
  end

  def <(other : Jug)
    LL.lt(self, other)
  end

  def <(other : Number)
    LL.lt(self, other)
  end

  def <=(other : Jug)
    LL.le(self, other)
  end

  def <=(other : Number)
    LL.le(self, other)
  end

  def ==(other : Jug)
    LL.eq(self, other)
  end

  def ==(other : Number)
    LL.eq(self, other)
  end
end
