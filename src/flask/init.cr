class Flask(T)
  getter data : Slice(T)
  getter size : Int32
  getter stride : Int32

  def to_s(io)
    io << "flask[" << @data.join(", ") << "]"
  end

  def initialize(data : Indexable(T))
    @size = data.size
    @stride = 1
    @data = Slice(T).new(size).copy_from(data)
  end

end

f = Flask.new [1, 2, 3]
