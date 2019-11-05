require "../core/ndtensor"

module Bottle::Testing
  extend self

  def tensor_equal(a, b)
    if a.size != b.size
      return false
    end
    a.size.times do |i|
      return false unless a[i] == b[i]
    end
    true
  end

  def row_a(a, b)
    a.nrows == b.nrows
  end

  def col_a(a, b)
    a.ncols == b.ncols
  end

  def matrix_aligned(a, b)
    row_a(a, b) && col_a(a, b)
  end

  def broadcast_columns_first(a, b)
    row_a(a, b) && a.ncols == 1
  end

  def broadcast_columns_second(a, b)
    row_a(a, b) && b.ncols == 1
  end

  def broadcast_rows_first(a, b)
    col_a(a, b) && a.nrows == 1
  end

  def broadcast_rows_second(a, b)
    col_a(a, b) && b.nrows == 1
  end
end
