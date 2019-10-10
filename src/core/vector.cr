require "../llib/lib_gsl"
require "../util/indexing"
require "../util/arithmetic"
require "../util/statistics"

class Vector
  include Bottle::Util::Indexing
  include Bottle::Util::Arithmetic

  @ptr : Pointer(LibGsl::GslVector)
  @size : Int32
  @vec : LibGsl::GslVector

  getter ptr

  def initialize(data : Indexable(Float64))
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
    return Vector(Float64).new(ptv.value, @size)
  end

  def +(other : Vector)
    _add_vec(self.copy, other)
  end

  def +(other : Number)
    _add_vec_constant(self.copy, other)
  end

  def -(other : Vector)
    _sub_vec(self.copy, other)
  end

  def -(other : Number)
    _sub_vec_constant(self.copy, other)
  end

  def *(other : Vector)
    _mul_vec(self.copy, other)
  end

  def *(other : Number)
    _mul_vec_constant(self.copy, other)
  end

  def /(other : Vector)
    _div_vec(self.copy, other)
  end

  def /(other : Number)
    _div_vec_constant(self.copy, other)
  end

  def [](ii : Int32)
    _take_vec_at_index(@ptr, ii)
  end

  def [](ii : Iterable(Int32))
    _take_vec_at_indexes(@ptr, ii)
  end

  def [](ii : Range(Int32, Int32))
    _take_vec_at_range(@ptr, ii)
  end

  def []=(ii : Int32, vv : Number)
    _set_vec_at_index(@ptr, ii, vv)
  end

  def max
    _vec_max(@ptr)
  end

  def min
    _vec_min(@ptr)
  end

  def ptpv
    _vec_ptpv(@ptr)
  end

  def ptp
    _vec_ptp(@ptr)
  end

  def idxmax
    _vec_idxmax(@ptr)
  end

  def idxmin
    _vec_idxmin(@ptr)
  end

  def ptpidx
    _vec_ptpidx(@ptr)
  end

  def to_s(io)
    io << "Vector[" << @vec.data.to_slice(@size).join(", ") << "]"
  end
end
