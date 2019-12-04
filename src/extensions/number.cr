require "../core/math"
require "../base/base"
require "../tensor/tensor"

macro extend_crystal_arithmetic(dtypes, operators)
  {% for dtype in dtypes %}
  struct {{dtype}}
    {% for operator in operators %}
      def {{operator[:sym].id}}(other : Bottle::BaseArray)
        Bottle::BMath.{{operator[:name]}}(self, other)
      end
    {% end %}
  end
  {% end %}
end

extend_crystal_arithmetic [Int32, Int8, Int16, Float32, Float64, Complex, UInt8, UInt16, UInt32], [
  {sym: :+, name: add},
  {sym: :-, name: subtract},
  {sym: :*, name: multiply},
  {sym: :/, name: divide},
  {sym: :**, name: power},
  {sym: ://, name: floordiv},
  {sym: :%, name: modulo},
  {sym: :==, name: equal},
  {sym: :>, name: greater},
  {sym: :>=, name: greater_equal},
  {sym: :<, name: less},
  {sym: :<=, name: less_equal},
]
