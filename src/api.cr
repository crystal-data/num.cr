require "./nd/binary"
require "./nd/ufunc"
require "./nd/ndtensor"
require "./nd/numeric"
require "./nd/statistics"

module Bottle::B
  extend self
  include Bottle::Internal::Binary
  include Bottle::Internal::UFunc
  include Bottle::Internal::Numeric
  include Bottle::Internal::Statistics
end
