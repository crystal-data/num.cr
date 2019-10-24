require "../util/exceptions"

abstract class Bottle::Internal::BottleObject(T)
  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Int32 || T == Bool %}
      {% raise "Wrong data type: #{T}. Types supported by Bottle are: #{SUPPORTED_TYPES}" %}
    {% end %}
  end

  protected def check_sign(n : Int32)
    unless n >= 0
      raise "Bottle does not support indexing with negative numbers"
    end
  end

  protected def range_to_slice(range, size)
    if !range.excludes_end?
      raise Bottle::Core::Exceptions::RangeError.new("Vectors do not support indexing with inclusive ranges. Always use '...'")
    end
    i = range.begin
    j = range.end
    start = (i.nil? ? 0 : i).as(UInt64 | Int32).to_i32
    finish = (j.nil? ? size : j).as(UInt64 | Int32).to_i32
    {start, finish}
  end

  protected def check_indexer(x, y)
    if x.size != y.size
      raise "Indexers must be the same size"
    end
  end

  protected def check_indexer(x, y, z)
    if x.size != y.size && y.size != z.size
      raise "Indexers must be the same size"
    end
  end
end
