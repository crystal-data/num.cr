def index_to_range(indexer, shape, strides)

  sh_ = [] of Int32
  st_ = [] of Int32

  ixdx = indexer.map_with_index do |idx, i|
    if idx.is_a?(Int32)
      idx
    else
      st_ << strides[i]
      x, y = Indexable.range_to_index_and_count(idx, shape[i])
      sh_ << y
      x
    end
  end

  {ixdx, sh_, st_}
end
