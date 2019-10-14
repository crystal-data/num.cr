require "../llib/gsl"
require "../util/exceptions"
require "../core/vector"

macro mask_comparison_abstract(type_, dtype, prefix)
  module Bottle::Ma::Comparison
    include Bottle::Util::Exceptions
    extend self
  end
end
