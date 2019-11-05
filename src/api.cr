require "./core/binary"
require "./core/ufunc"
require "./core/ndtensor"
require "./core/numeric"
require "./core/statistics"
require "./core/trig"
require "./linalg/fixed_dimension"
require "./core/assemble"
require "./core/printoptions"

module Bottle::B
  extend self
  include Bottle::Internal::Binary
  include Bottle::Internal::UFunc
  include Bottle::Internal::Numeric
  include Bottle::Internal::Statistics
  include Bottle::Internal::Trigonometric
  include Bottle::Internal::LinAlg
  include Bottle::Internal::Assemble
  include Bottle::Internal::ToString
end
