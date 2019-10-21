require "../core/flask"
require "../core/jug"

module Bottle::Testing
  extend self

  def flask_equal(a, b)
    if a.size != b.size
      return false
    end
    return (a == b).all?
  end
end
