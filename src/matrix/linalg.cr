require "../core/matrix/*"
require "./*"
require "../core/vector/*"

class Matrix(T, D)
  def inv
    return Bottle::Core::Linalg.matrix_inverse(self.copy)
  end

  def tril(k = 0)
    m = self.copy
    (0...nrows).each do |i|
      (0...ncols).each do |j|
        if i < j - k
          m[i, j] = 0.0
        end
      end
    end
    return m
  end

  def triu(k = 0)
    m = self.copy
    (0...nrows).each do |i|
      (0...ncols).each do |j|
        if i > j - k
          m[i, j] = 0.0
        end
      end
    end
    return m
  end
end
