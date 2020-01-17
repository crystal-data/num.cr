# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
require "../base"
require "./format"
require "../core/assemble"

module NumInternal
  extend self

  # For arrays that are too large to display the entire data structure,
  # this method slices the proper regions to find the correct pad value
  # for the region of the array that will be displayed
  private def leading_trailing(a : Num::BaseArray, edgeitems : Int32, index = [] of Range(Int32?, Int32?))
    axis = index.size
    if axis == a.ndims
      return a.slice(index)
    end
    if a.shape[axis] > 2 * edgeitems
      Num.concatenate(
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

  # Finds the maximum width of a string representation of an element of
  # an ndarray
  private def max_width(a : Num::BaseArray)
    mx = 0
    a.iter.each do |i|
      val = format(i.value)
      if val.size > mx
        mx = val.size
      end
    end
    mx
  end

  # Extends a line if possible, adding in proper wrapping for areas
  # where the line would overflow the maximum line width
  private def extend_line(s : String, line : String, word : String, line_width : Int32, next_line_prefix : String)
    needs_wrap = line.size + word.size > line_width

    # If the line is too long, the entire element will be sent
    # to the next line to avoid lines overlapping
    if needs_wrap
      s += line.rstrip + "\n"
      line = next_line_prefix
    end
    line += word
    {s, line}
  end

  # Don't even try to read this, it's bad lol, but it works
  private def recursor(a, index, hanging_indent, curr_width, summary_insert, edge_items, separator = ", ", pad = 0)
    axis = index.size
    axes_left = a.ndims - axis

    if axes_left == 0
      return format(a.slice(index).value, pad)
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
        word = recursor(a, index + [i], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
        s, line = extend_line(s, line, word, elem_width, hanging_indent)
        line += separator
      end

      if show_summary
        s, line = extend_line(s, line, summary_insert, elem_width, hanging_indent)
        line += separator
      end

      (trailing_items).step(to: 2, by: -1) do |i|
        word = recursor(a, index + [-i], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
        s, line = extend_line(s, line, word, elem_width, hanging_indent)
        line += separator
      end

      word = recursor(a, index + [-1], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
      s, line = extend_line(s, line, word, elem_width, hanging_indent)
      s += line
    else
      s = ""
      line_sep = separator.rstrip + "\n" * (axes_left - 1)

      leading_items.times do |i|
        nested = recursor(a, index + [i], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
        s += hanging_indent + nested + line_sep
      end

      if show_summary
        s += hanging_indent + summary_insert + ", \n"
      end

      (trailing_items).step(to: 2, by: -1) do |i|
        nested = recursor(a, index + [-i], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
        s += hanging_indent + nested + line_sep
      end

      nested = recursor(a, index + [-1], next_hanging_indent, next_width, summary_insert, edge_items, separator, pad)
      s += hanging_indent + nested
    end
    "[" + s[hanging_indent.size...] + "]"
  end

  # Wrapper around recursor
  def format_array(a, line_width, next_line_prefix, separator, edge_items, summary_insert, pad)
    recursor(a, [] of Int32, hanging_indent: next_line_prefix, curr_width: line_width, summary_insert: summary_insert, edge_items: edge_items, separator: separator, pad: pad)
  end

  # Converts an array to its string reprensentation
  def array_to_string(a : Num::BaseArray, separator = ", ", prefix = "", edge_items = 3, summary_insert = "...", line_width = 75)
    if a.size > 1000
      data = leading_trailing(a, edge_items)
    else
      summary_insert = ""
      data = a
    end

    pad = max_width(data)
    next_line_prefix = " "
    next_line_prefix += " " * prefix.size
    format_array(a, line_width, next_line_prefix, separator, edge_items, summary_insert, pad)
  end
end
