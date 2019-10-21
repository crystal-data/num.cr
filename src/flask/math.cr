require "../libs/dtype"
require "./*"
require "../math/*"

class Flask(T)
  # Elementwise addition of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other : Flask(T))
    LL.add(self.clone, other)
  end

  # Elementwise addition of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [3.0, 4.0, 5.0]
  # ```
  def +(other : T)
    LL.add(self.clone, other)
  end

  # Elementwise subtraction of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other : Flask(T))
    LL.sub(self.clone, other)
  end

  # Elementwise subtraction of a Flask with a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 - f2 # => [-1.0, 0.0, 1.0]
  # ```
  def -(other : T)
    LL.sub(self.clone, other)
  end

  # Elementwise multiplication of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other : Flask(T))
    LL.mul(self.clone, other)
  end

  # Elementwise multiplication of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 + f2 # => [2.0, 4.0, 6.0]
  # ```
  def *(other : T)
    LL.mul(self.clone, other)
  end

  # Elementwise division of a Flask to another equally sized Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other : Flask(T))
    LL.div(self.clone, other)
  end

  # Elementwise division of a Flask to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2
  # f1 / f2 # => [0.5, 1, 1.5]
  # ```
  def /(other : T)
    LL.div(self.clone, other)
  end

  # Elementwise greater than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 > f2 # => [false, false, true ]
  # ```
  def >(other : Flask)
    LL.gt(self, other)
  end

  # Elementwise greater than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [false, false, true ]
  # ```
  def >(other : Number)
    LL.gt(self, other)
  end

  # Elementwise greater equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 >= f2 # => [false, true, true ]
  # ```
  def >=(other : Flask)
    LL.ge(self, other)
  end

  # Elementwise greater equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 >= f2 # => [false, true, true ]
  # ```
  def >=(other : Number)
    LL.ge(self, other)
  end

  # Elementwise less than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 < f2 # => [true, false, false]
  # ```
  def <(other : Flask)
    LL.lt(self, other)
  end

  # Elementwise less than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 < f2 # => [false, false, true ]
  # ```
  def <(other : Number)
    LL.lt(self, other)
  end

  # Elementwise less equal than comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 <= f2 # => [true, true, false]
  # ```
  def <=(other : Flask)
    LL.le(self, other)
  end

  # Elementwise less equal than comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [true, true, false]
  # ```
  def <=(other : Number)
    LL.le(self, other)
  end

  # Elementwise equal comparison to another Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = Flask.new [3.0, 2.0, 1.0]
  # f1 == f2 # => [false, true, false ]
  # ```
  def ==(other : Flask)
    LL.eq(self, other)
  end

  # Elementwise equal comparison to a scalar
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f2 = 2.0
  # f1 > f2 # => [false, true, false]
  # ```
  def ==(other : Number)
    LL.eq(self, other)
  end

  # Sum reduction for a Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 3.0]
  # f1.sum # => 6.0
  # ```
  def sum
    reduce { |i, j| i + j }
  end

  # Product reduction for a Flask
  #
  # ```
  # f1 = Flask.new [1.0, 2.0, 4.0]
  # f1.prod # => 8.0
  # ```
  def prod
    reduce { |i, j| i * j }
  end
end
