require "../libs/gsl"
require "../core/vector/*"
require "./*"

class Vector(T, D)
  @ptr : Pointer(T)
  @obj : T
  @owner : Int32
  @size : UInt64
  @stride : UInt64
  @data : Pointer(D)
  @dtype : D.class

  getter ptr
  getter owner
  getter size
  getter stride
  getter data
  getter dtype

  def self.new(data : Array(A)) forall A
    new(*fetch_struct(data))
  end

  private def self.fetch_struct(items : Array(Int32))
    vector = LibGsl.gsl_vector_int_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_int_set(vector, i, e) }
    return vector, vector.value.data
  end

  private def self.fetch_struct(items : Array(Float64))
    vector = LibGsl.gsl_vector_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
    return vector, vector.value.data
  end

  private def self.fetch_struct(items : Array(Float64 | Int32))
    vector = LibGsl.gsl_vector_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
    return vector, vector.value.data
  end

  def initialize(@ptr : Pointer(T), @data : Pointer(D))
    @obj = @ptr.value
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
    @dtype = D
  end

  def initialize(@obj : T, @data : Pointer(D))
    @ptr = pointerof(@obj)
    @owner = @obj.owner
    @size = @obj.size
    @stride = @obj.stride
    @dtype = D
  end

  def self.zeros(n : Int32, dtype : Float64.class)
    vector = LibGsl.gsl_vector_calloc(n)
    return Vector.new vector, vector.value.data
  end

  def self.zeros(n : Int32, dtype : Int32.class)
    vector = LibGsl.gsl_vector_int_calloc(n)
    return Vector.new vector, vector.value.data
  end
end
