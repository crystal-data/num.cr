require "../src/num"

a = Tensor.new([100000000]) { |i| i }
b = Tensor.new([100000000]) { |i| i }
ret = Num.empty([100000000], dtype: Int32)

ch_in = Channel({Int32, Int32, Int32}).new
ch_out = Channel({Int32, Int32}).new

8.times do
  spawn do
    loop do
      x, y, i = ch_in.receive
      ch_out.send({x + y, i})
    end
  end
end

iter_a = a.unsafe_iter
iter_b = b.unsafe_iter
buf = ret.buffer

spawn do
  a.size.times do |i|
    ch_in.send({iter_a.next.value, iter_b.next.value, i})
  end
end

ret.size.times do
  val, index = ch_out.receive
  buf[index] = val
end

puts ret
