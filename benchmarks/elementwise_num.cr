require "../src/num"

def elementwise(n : Int32)
  puts "\nNum.cr elementwise #{n}  1.0 + i / 1.0 + j * k"
  i = Tensor.random(0.0...1.0, [n])
  j = Tensor.random(0.0...1.0, [n])
  k = Tensor.random(0.0...1.0, [n])
  i.map3(j, k) { |i, j, k| 1.0 + i / 1.0 + j * k }
end

n = (ARGV[0]? || 100).to_i
c = elementwise(n)
puts c[n//2]
