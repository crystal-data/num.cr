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
    vector = LibGsl.gsl_vector_alloc(items.size)
    items.each_with_index { |e, i| LibGsl.gsl_vector_set(vector, i, e) }
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

  def to_s(io)
    vals = (0...@size).map { |i| Bottle::Core::VectorIndex.get_vector_element_at_index(@ptr, i) }
    io << "[" << vals.map {|v| v.round(3) }.join(", ") << "]"
  end

  def self.zeros(n : Int32 | UInt64)
    vector = LibGsl.gsl_vector_calloc(n)
    return Vector.new vector, vector.value.data
  end

  def self.empty(n : Int32 | UInt64)
    vector = LibGsl.gsl_vector_alloc(n)
    return Vector.new vector, vector.value.data
  end

  def self.random(n : Int32 | UInt64)
    Vector.new (0...n).map { |_| Random.rand }
  end
end
