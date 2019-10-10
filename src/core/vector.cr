require "../llib/lib_gsl"
require "benchmark"

class BottleObject(T)
  @ptr : Pointer(LibGsl::GslVector)
  @size : Int32
  @vec : LibGsl::GslVector

  def initialize(data : Indexable(T))
    ptv = LibGsl.gsl_vector_alloc(data.size)
    data.each_with_index { |e, i| LibGsl.gsl_vector_set(ptv, i, e) }
    @ptr = ptv
    @size = data.size
    @vec = @ptr.value
  end

  def initialize(vec : LibGsl::GslVector, @size)
    @vec = vec
    @ptr = pointerof(@vec)
  end

  def copy
    ptv = LibGsl.gsl_vector_alloc(@size)
    LibGsl.gsl_vector_memcpy(ptv, @ptr)
    return BottleObject(T).new(ptv.value, @size)
  end

  def +(other : BottleObject)
    ret = self.copy
    LibGsl.gsl_vector_add(ret.@ptr, other.@ptr)
    return ret
  end

  def +(other : Number)
    LibGsl.gsl_vector_add_constant(@ptr, other)
  end

  def -(other : BottleObject)
    ret = self.copy
    LibGsl.gsl_vector_sub(ret.@ptr, other.@ptr)
    return ret
  end

  def -(other : Number)
    ret = self.copy
    LibGsl.gsl_vector_add_constant(ret.@ptr, -other)
    return ret
  end

  def *(other : BottleObject)
    ret = self.copy
    LibGsl.gsl_vector_mul(ret.@ptr, other.@ptr)
    return ret
  end

  def *(other : Number)
    ret = self.copy
    LibGsl.gsl_vector_scale(ret.@ptr, other)
    return ret
  end

  def /(other : BottleObject)
    ret = self.copy
    LibGsl.gsl_vector_div(ret.@ptr, other.@ptr)
    return ret
  end

  def /(other : Number)
    ret = self.copy
    LibGsl.gsl_vector_scale(ret.@ptr, 1 / other)
    return ret
  end

  def [](ii : Int32)
    return LibGsl.gsl_vector_get(@ptr, ii)
  end

  def [](ii : Iterable(Int32))
    return ii.map { |e| LibGsl.gsl_vector_get(@ptr, e) }
  end

  def [](ii : Range(Int32, Int32))
    start = ii.begin
    close = ii.excludes_end? ? ii.end - 1 : ii.end
    view = LibGsl.gsl_vector_subvector(@ptr, ii.begin, close - ii.begin)
    return BottleObject(T).new(view.vector, close - ii.begin)
  end

  def []=(ii : Int32, vv : T)
    LibGsl.gsl_vector_set(@ptr, ii, vv)
  end

  def max
    return LibGsl.gsl_vector_max(@ptr)
  end

  def min
    return LibGsl.gsl_vector_min(@ptr)
  end

  def ptpv
    min_out = 0.0
    max_out = 0.0
    LibGsl.gsl_vector_minmax(@ptr, pointerof(min_out), pointerof(max_out))
    return min_out, max_out
  end

  def ptp
    mn, mx = self.ptpv
    return mx - mn
  end

  def idxmax
    return LibGsl.gsl_vector_max_index(@ptr)
  end

  def idxmin
    return LibGsl.gsl_vector_min_index(@ptr)
  end

  def ptpidx
    imin : UInt64 = 0
    imax : UInt64 = 0
    LibGsl.gsl_vector_minmax_index(@ptr, pointerof(imin), pointerof(imax))
    return imin, imax
  end

  def to_s(io)
    io << "Vector[" << @vec.data.to_slice(@size).join(", ") << "]"
  end
end
