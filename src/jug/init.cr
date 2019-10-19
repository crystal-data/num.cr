require "./*"
require "../libs/dtype"
require "../flask/*"

class Jug(T)
  getter data : Slice(T)
  getter nrows : Int32
  getter ncols : Int32
  getter xstride : Int32
  getter ystride : Int32

  def initialize(data : Indexable(Indexable(T)))
    @nrows = data.size
    @ncols = data[0].size
    @ystride = ncols
    @xstride = 1
    @data = Slice(T).new(nrows * ncols) do |idx|
      i = idx // nrows
      j = idx % nrows
      data[i][j]
    end
  end

  def initialize(@data, @nrows, @ncols, @xstride, @ystride)
  end

  def [](index : Indexer)
    Flask.new data[index * ncols, ncols * xstride], ncols, xstride
  end

  def [](rng : Range(Nil, Nil), column : Indexer)
    Flask.new data[column, nrows * ystride - column], nrows, ystride
  end

  def to_s(io)
    io << "["
    (0...@nrows).each do |el|
      startl = el == 0 ? "" : " "
      endl = el == @nrows - 1 ? "" : "\n"
      row = self[el]
      io << startl << row << endl
    end
    io << "]"
  end
end
