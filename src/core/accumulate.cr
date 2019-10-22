require "./flask"

class Accumulate(T)
  getter data : Flask(T)
  getter size : Int32
  getter inplace : Bool

  def initialize(data : Flask(T), @inplace = false)
    @size = data.size
    if inplace
      @data = data.clone
    else
      @data = data
    end
  end

  def sum
    (1...size).each do |i|
      data[i] += data[i - 1]
    end
    return data unless inplace
  end

  def prod
    (1...size).each do |i|
      data[i] *= data[i - 1]
    end
    return data unless inplace
  end

  def min
    (1...size).each do |i|
      data[i] = Math.min(data[i], data[i - 1])
    end
    return data unless inplace
  end

  def max
    (1...size).each do |i|
      data[i] = Math.max(data[i], data[i - 1])
    end
    return data unless inplace
  end
end

f = Flask.new [1, 2, 3]
puts f.accumulate.max
