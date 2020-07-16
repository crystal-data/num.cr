require "spec"
require "../src/num"

def assert_array_equal(a, b)
  Num.all_close(a, b).should be_true
end
