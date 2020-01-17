require "spec"
require "../src/num"

def assert_array_equal(a, b)
  Num.allclose(a, b).should be_true
end
