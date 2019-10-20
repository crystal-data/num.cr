require "../libs/dtype"
require "../libs/cblas"
require "../flask/*"
require "../jug/*"

macro blas_helper(dtype, blas_prefix, cast)
  module LL
    extend self

  end
end
