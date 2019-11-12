require "./arrayops/binary"
require "./arrayops/build"
require "./arrayops/math"
require "./arrayops/trig"
require "./tensor/tensor"
require "./linalg/fixed_dimension"
require "./linalg/reductions"
require "./tensor/creation"
require "./chararray/chararray"
require "./mask/mask"

module Bottle::B
  extend self
  include Bottle::Binary
  include Bottle::BMath
  include Bottle::Creation
  include Bottle::Statistics
  include Bottle::Trigonometry
  include Bottle::LinAlg
  include Bottle::Assemble
end
