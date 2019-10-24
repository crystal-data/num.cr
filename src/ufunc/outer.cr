require "../core/vector"
require "../core/matrix"

class Bottle::UFunc::Outer(T)
  getter it : Vector(T)
  getter other : Vector(T)

  def initialize(@it : Vector(T), @other : Vector(T))
  end

  def outer(&block)
    Matrix(T).new(it.size, other.size) do |i, j|
      yield i, j
    end
  end

  def add
    outer do |i, j|
      it[i] + other[j]
    end
  end

  def subtract
    outer do |i, j|
      it[i] - other[j]
    end
  end

  def multiply
    outer do |i, j|
      it[i] * other[j]
    end
  end

  def divide
    outer do |i, j|
      it[i] / other[j]
    end
  end

  def minimum
    outer do |i, j|
      Math.min(it[i], other[j])
    end
  end

  def maximum
    outer do |i, j|
      Math.max(it[i], other[j])
    end
  end
end
