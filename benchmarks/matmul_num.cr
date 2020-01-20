require "../src/num"

def matmul(n : Int32)
  puts "\nNum.cr matmul #{n}x#{n}"
  a = Tensor.random(0.0...1.0, [n, n])
  b = Tensor.random(0.0...1.0, [n, n])
  a.matmul(b)
end

n = (ARGV[0]? || 100).to_i
c = matmul(n)
puts c[n//2, n//2]
