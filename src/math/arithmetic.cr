require "../flask/*"
require "../jug/*"

module LL
  extend self

  def add(a : Flask, b : Flask)
    a.each_index { |i| a[i] += b[i] }
    return a
  end

  def add(a : Jug, b : Jug)
    a.each_index { |i, j| a[i, j] += b[i, j] }
    return a
  end

  def add(a : Flask, x : Number)
    a.each_index { |i| a[i] += x }
  end

  def add(a : Jug, x : Number)
    a.each_index { |i, j| a[i, j] += x }
    return a
  end

  def sub(a : Flask, b : Flask)
    a.each_index { |i| a[i] -= b[i] }
    return a
  end

  def sub(a : Jug, b : Jug)
    a.each_index { |i, j| a[i, j] -= b[i, j] }
    return a
  end

  def sub(a : Flask, x : Number)
    a.each_index { |i| a[i] -= x }
    return a
  end

  def sub(a : Jug, x : Number)
    a.each_index { |i, j| a[i, j] -= x }
    return a
  end

  def mul(a : Flask, b : Flask)
    a.each_index { |i| a[i] *= b[i] }
    return a
  end

  def mul(a : Jug, b : Jug)
    a.each_index { |i, j| a[i, j] *= b[i, j] }
    return a
  end

  def mul(a : Flask, x : Number)
    a.each_index { |i| a[i] *= x }
    return a
  end

  def mul(a : Jug, x : Number)
    a.each_index { |i, j| a[i, j] *= x }
    return a
  end

  def div(a : Flask, b : Flask)
    a.each_index { |i| a[i] /= b[i] }
    return a
  end

  def div(a : Jug, b : Jug)
    a.each_index { |i, j| a[i, j] /= b[i, j] }
    return a
  end

  def div(a : Flask, x : Number)
    a.each_index { |i| a[i] /= x }
    return a
  end

  def div(a : Jug, x : Number)
    a.each_index { |i, j| a[i, j] /= x }
    return a
  end
end
