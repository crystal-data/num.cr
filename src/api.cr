require "./core/binary"
require "./core/fromnumeric"
require "./core/numeric"
require "./core/trig"
require "./core/ufunc"

module Bottle::B
  include Bottle::Internal::Binary
  include Bottle::Internal::FromNumeric
  include Bottle::Internal::Numeric
  include Bottle::Internal::Trigonometric
  include Bottle::Internal::UFunc
end
