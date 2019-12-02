require "./tensor/tensor"
require "./tensor/linalg"
require "./fft/real"
require "./financial/simple"
require "./tensor/creation"
require "./sparse/coo"
require "./core/assemble"
require "./core/converters"
require "./core/math"
require "./core/reductions"
require "./core/search"
require "./io"
require "./extensions/number"

module Bottle::B
  extend self
  include Bottle::BMath
  include Bottle::Creation
  include Bottle::Statistics
  include Bottle::Assemble
  include Bottle::InputOutput
  include Bottle::Search
  include Bottle::FFT
  include Bottle::Financial
  include Bottle::Sparse
end
