require "./core/binary"
require "./core/ufunc"
require "./core/ndtensor"
require "./core/numeric"
require "./core/statistics"
require "./core/trig"
require "./linalg/fixed_dimension"

module Bottle::B
  extend self
  include Bottle::Internal::Binary
  include Bottle::Internal::UFunc
  include Bottle::Internal::Numeric
  include Bottle::Internal::Statistics
  include Bottle::Internal::Trigonometric
  include Bottle::Internal::LinAlg
end
