require "../src/num"

def elementwise(n : Int32)
  puts "\nNum.cr elementwise #{n}  1.0 + i / 1.0 + j * k"
  x = Tensor.random(0.0...1.0, [n])
  y = Tensor.random(0.0...1.0, [n])
  z = Tensor.random(0.0...1.0, [n])
  x.map3(y, z) { |i, j, k| 1.0 + i / 1.0 + j * k }
end

n = (ARGV[0]? || 100).to_i
c = elementwise(n)
puts c[n//2]
