require "./macros"

module Bottle::Internal::Core
  elementwise(:+, add)
  elementwise(:-, subtract)
  elementwise(:*, multiply)
  elementwise(:/, divide)
  elementwise(:**, power)
  elementwise(://, floordiv)
  elementwise(:%, modulo)
  elementwise(:==, equal)
  elementwise(:>, greater)
  elementwise(:>=, greater_equal)
  elementwise(:<, less)
  elementwise(:<=, less_equal)

  elementwise(:&, bitwise_and)
  elementwise(:|, bitwise_or)
  elementwise(:^, bitwise_xor)
  elementwise(:<<, left_shift)
  elementwise(:>>, right_shift)
end
