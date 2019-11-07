require "./core/binary"
require "./core/ufunc"
require "./core/ndtensor"
require "./core/numeric"
require "./core/statistics"
require "./core/trig"
require "./linalg/fixed_dimension"
require "./core/assemble"
require "./core/printoptions"
require "./util/testing"

module Bottle::B
  extend self
  include Bottle::Internal
end
