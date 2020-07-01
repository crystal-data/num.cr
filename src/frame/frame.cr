require "../api"
require "./series"
require "./internal/print"

class DataFrame(*T, V)
  @data : *T
  getter columns : Hash(V, Int32)
  getter size : Int32

  def data
    @data
  end

  def initialize(*args : *T, columns : Hash(V, Int32))
    @data = args
    @size = args[0].data.size
    @columns = columns
  end

  def self.from_items(*args)
    args = args_to_series(*args)
    columns = Hash(Int32, Int32).new
    args.size.times do |a|
      columns[a] = a
    end
    new(*args, columns: columns)
  end

  def self.from_data(*args, columns : Array(U)) forall U
    args = args_to_series(*args)
    cols = Hash(U, Int32).new
    columns.each_with_index do |c, i|
      cols[c] = i
    end
    new(*args, columns: cols)
  end

  def [](k : V, d : U.class) forall U
    n = @columns[k]
    @data[n].as(Series(U))
  end

  def to_s(io)
    io << Num::Internal.print_df(self)
  end

  private def self.args_to_series(*args : *U) forall U
    {% begin %}
      Tuple.new(
        {% for i in 0...U.size %}
          Series.new(args[{{i}}]),
        {% end %}
      )
    {% end %}
  end

  macro map_macro(fn, cb)
    def {{fn.id}}
      \{% if true %}
        Tuple.new(
          \{% for i in 0...T.size %}
            {{cb.id}}(@data[\{{i}}]),
          \{% end %}
        )
      \{% end %}
    end
  end
end
