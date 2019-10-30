def indexes(shape : Array(Int32))
  Array.each_product(
    shape.map { |i| (0...i).to_a }) { |p| yield p }
end

def newline(shape, index)
  diff = shape.zip(index).map { |i, j| i - j }
  nl = 0
  (1..diff.size).each do |i|
    break unless diff[-i] == 1
    nl += 1
  end
  if nl == shape.size
    ("]" * nl)
  else
    ("]" * nl) + ("\n" * Math.min(nl, 2))
  end
end

def startline(shape, index)
  diff = shape.zip(index).map { |i, j| (i - j) == i }
  sl = 0
  (1..diff.size).each do |i|
    break unless diff[-i]
    sl += 1
  end
  "[" * sl
end
