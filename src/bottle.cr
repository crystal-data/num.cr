require "./core/flask"
require "./core/jug"

module Bottle
  extend self
  VERSION = "0.1.1"
end

f = Flask.random(0...10, 5)
puts f
