require "../core/vector"

module Bottle::Testing
  extend self

  def vector_equal(a, b)
    if a.size != b.size
      return false
    end
    a.size.times do |i|
      return false unless a[i] == b[i]
    end
    true
  end
end
