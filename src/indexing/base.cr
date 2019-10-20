require "../core/exceptions"

module LL
  extend self
  # Normalizes a slice range to allow indexing to work with gsl.  This means
  # bounding nil ranges and disallowing inclusive ranges
  def convert_range_to_slice(range : Range(Int32 | Nil, Int32 | Nil), size : Int32)
    if !range.excludes_end?
      raise Bottle::Core::Exceptions::RangeError.new("Vectors do not support indexing with inclusive ranges. Always use '...'")
    end
    i = range.begin
    j = range.end
    start = (i.nil? ? 0 : i).as(UInt64 | Int32).to_u64
    finish = (j.nil? ? size : j).as(UInt64 | Int32).to_u64
    start...finish
  end
end
