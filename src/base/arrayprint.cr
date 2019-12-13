require "./base"

module Num::ArrayPrint
  extend self

  class IntFormatter
    getter pad : Int32

    def initialize(a : BaseArray)
      @pad = max_sl(a)
    end

    private def max_sl(a)
      mx = ""
      a.flat_iter.each do |i|
        s = "#{i.value}"
        if s.size > mx.size
          mx = s
        end
      end
      mx.size
    end

    def format(n)
      "#{n}".rjust(pad)
    end
  end

  class FloatFormatter
    getter pad : Int32

    def initialize(a : BaseArray)
      @pad = max_sl(a)
    end

    private def max_sl(a)
      mx = ""
      a.flat_iter.each do |i|
        s = "#{i.value.round(4)}"
        if s.size > mx.size
          mx = s
        end
      end
      mx.size
    end

    def format(n)
      "#{n.round(4)}".ljust(pad)
    end
  end

  private def get_formatter(a : BaseArray(Float32) | BaseArray(Float64))
    FloatFormatter.new(a)
  end

  private def get_formatter(a : BaseArray)
    IntFormatter.new(a)
  end

  private def leading_trailing(a, edgeitems, index = [] of Range(Int32?, Int32?))
    axis = index.size
    if axis == a.ndims
      return a.slice(index)
    end
    if a.shape[axis] > 2 * edgeitems
      N.concatenate(
        [
          leading_trailing(a, edgeitems, index + [...edgeitems]),
          leading_trailing(a, edgeitems, index + [(-1 * edgeitems)...]),
        ],
        axis: axis
      )
    else
      leading_trailing(a, edgeitems, index + [...])
    end
  end

  def array2string(a, separator = ", ", prefix = "")
    a = a.dup('C')
    if a.size > 1000
      summary_insert = "..."
      data = leading_trailing(a, 3)
    else
      summary_insert = ""
      data = a
    end
    formatter = get_formatter(data)
    next_line_prefix = " "
    next_line_prefix += " " * prefix.size
    format_array(a, 75, next_line_prefix, separator, 3, summary_insert, formatter: formatter)
  end

  private def _extend_line(s, line, word, line_width, next_line_prefix)
    needs_wrap = line.size + word.size > line_width

    if needs_wrap
      s += line.rstrip + "\n"
      line = next_line_prefix
    end
    line += word
    {s, line}
  end

  private def recursor(a, index, hanging_indent, curr_width, summary_insert, edge_items, formatter, separator = ", ")
    axis = index.size
    axes_left = a.ndims - axis

    if axes_left == 0
      return formatter.format(a.slice(index).value)
    end

    next_hanging_indent = hanging_indent + ' '
    next_width = curr_width - 1

    a_len = a.shape[axis]
    show_summary = (summary_insert.size > 0) && (2 * edge_items < a_len)
    if show_summary
      leading_items = edge_items
      trailing_items = edge_items
    else
      leading_items = 0
      trailing_items = a_len
    end

    s = ""

    if axes_left == 1
      elem_width = curr_width - {separator.rstrip.size, 1}.max
      line = hanging_indent

      leading_items.times do |i|
        word = recursor(a, index + [i], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
        s, line = _extend_line(s, line, word, elem_width, hanging_indent)
        line += separator
      end

      if show_summary
        s, line = _extend_line(s, line, summary_insert, elem_width, hanging_indent)
        line += separator
      end

      (trailing_items).step(to: 2, by: -1) do |i|
        word = recursor(a, index + [-i], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
        s, line = _extend_line(s, line, word, elem_width, hanging_indent)
        line += separator
      end

      word = recursor(a, index + [-1], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
      s, line = _extend_line(s, line, word, elem_width, hanging_indent)
      s += line
    else
      s = ""
      line_sep = separator.rstrip + "\n" * (axes_left - 1)

      leading_items.times do |i|
        nested = recursor(a, index + [i], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
        s += hanging_indent + nested + line_sep
      end

      if show_summary
        s += hanging_indent + summary_insert + ", \n"
      end

      (trailing_items).step(to: 2, by: -1) do |i|
        nested = recursor(a, index + [-i], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
        s += hanging_indent + nested + line_sep
      end

      nested = recursor(a, index + [-1], next_hanging_indent, next_width, summary_insert, edge_items, formatter, separator)
      s += hanging_indent + nested
    end
    "[" + s[hanging_indent.size...] + "]"
  end

  def format_array(a, line_width, next_line_prefix, separator, edge_items, summary_insert, formatter)
    recursor(a, [] of Int32, hanging_indent: next_line_prefix, curr_width: line_width, summary_insert: summary_insert, edge_items: edge_items, formatter: formatter, separator: separator)
  end
end
