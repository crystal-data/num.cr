require "./core/binary"
require "./core/fromnumeric"
require "./core/numeric"
require "./core/trig"
require "./core/ufunc"
require "./linalg/numeric"

module Bottle::B
  include Bottle::Internal::Binary
  include Bottle::Internal::FromNumeric
  include Bottle::Internal::Numeric
  include Bottle::Internal::Trigonometric
  include Bottle::Internal::UFunc
  include Bottle::Internal::LinAlg
end
