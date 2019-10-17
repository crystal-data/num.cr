require "./gsl"

@[Link(ldflags: "#{__DIR__}/../../ext/bottle.so")]
lib LibBottle
  fun gsl_vector_ma_equal(u : LibGsl::GslVector*, v : LibGsl::GslVector*) : LibGsl::GslVectorInt
end
