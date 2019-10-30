require "./nd/binary"
require "./nd/ufunc"
require "./nd/ndtensor"
require "./nd/numeric"

module Bottle::B
  extend self
  include Bottle::NDimensional::Binary
  include Bottle::NDimensional::UFunc
  include Bottle::NDimensional::Numeric
end
