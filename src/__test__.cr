require "./api"
require "./testing/testing"
require "./core/exceptions"
require "spec"
include Bottle
include Bottle::Internal
include Bottle::Testing
include Bottle::Exceptions

class MockArray(T) < BaseArray(T)
  def check_type
  end

  def basetype
    MockArray
  end
end
