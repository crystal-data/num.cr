require "./api"
require "./testing/testing"
require "./core/exceptions"
require "spec"
require "complex"
include Num
include Num::Internal
include Num::Testing
include Num::Exceptions

class MockArray(T) < BaseArray(T)
  def check_type
  end

  def basetype
    MockArray
  end
end
