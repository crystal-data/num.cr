require "./arrayops/binary"
require "./arrayops/build"
require "./arrayops/math"
require "./arrayops/trig"
require "./arrayops/io"
require "./tensor/tensor"
require "./linalg/fixed_dimension"
require "./linalg/reductions"
require "./tensor/creation"

module Bottle::B
  extend self
  include Bottle::Binary
  include Bottle::BMath
  include Bottle::Creation
  include Bottle::Statistics
  include Bottle::Trigonometry
  include Bottle::LinAlg
  include Bottle::Assemble
  include Bottle::InputOutput
end
