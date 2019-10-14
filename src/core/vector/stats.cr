require "../../libs/gsl"
require "../../vector/*"

module Bottle::Core::VectorStats
  extend self

  # Computes the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.max # => 4
  # ```
  def vector_max(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_max(a)
  end

  # Computes the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.min # => 1
  # ```
  def vector_min(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_min(a)
  end

  # Computes the min and max values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpv # => {1, 4}
  # ```
  def vector_ptpv(a : Pointer(LibGsl::GslVector))
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_vector_minmax(a, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  # Computes the "peak to peak" of a vector (max - min)
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def vector_ptp(a : Pointer(LibGsl::GslVector))
    mn, mx = vector_ptpv(a)
    return mx - mn
  end

  # Computes the index of the maximum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmax # => 3
  # ```
  def vector_idxmax(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_max_index(a)
  end

  # Computes the index of the minimum value of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.idxmin # => 0
  # ```
  def vector_idxmin(a : Pointer(LibGsl::GslVector))
    return LibGsl.gsl_vector_min_index(a)
  end

  # Computes the indexes of the minimum and maximum values of a vector
  #
  # ```
  # v = Vector.new [1, 2, 3, 4]
  # v.ptpidx # => {0, 3}
  # ```
  def vector_ptpidx(a : Pointer(LibGsl::GslVector))
    imin : UInt64 = 0
    imax : UInt64 = 0
    LibGsl.gsl_vector_minmax_index(a, pointerof(imin), pointerof(imax))
    return imin, imax
  end
end
