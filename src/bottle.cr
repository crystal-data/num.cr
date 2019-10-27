require "./core/tensor"
require "./core/matrix"
require "./api/*"
require "./linalg/*"
require "./blas/*"
require "./libs/*"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.1.1"
end
