require "../jug/*"
require "../flask/*"

module BottleTest
  extend self

  def flask_equal(a, b)
    if a.size != b.size
      return false
    end
    return (a == b).all?
  end
end
