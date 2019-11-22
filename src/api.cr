require "./arrayops/binary"
require "./arrayops/build"
require "./arrayops/math"
require "./arrayops/trig"
require "./arrayops/io"
require "./arrayops/search"
require "./tensor/tensor"
# require "./linalg/fixed_dimension"
# require "./linalg/reductions"
require "./linalg/products"
require "./linalg/decompositions"
require "./linalg/eigenvalues"
require "./linalg/norms"
require "./fft/real"
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
  include Bottle::Search
  include Bottle::FFT
end
