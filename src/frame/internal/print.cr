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

require "../../tensor/internal/print"
require "../series"

module Num::Internal
  def print_series(ser : Series)
    m = max_width(ser.data)
    s = ""
    ser.data.each do |e|
      s += format(e, m)
      s += "\n"
    end
    s += "name: #{ser.name}"
    s
  end

  def print_df(df : DataFrame)
    widths = df.data.map_with_index do |ser, i|
      {max_width(ser.data), "#{df.columns.keys[i]}".size}.max
    end

    df_size = df.size
    index_width = "#{df_size - 1}".size
    c = 2
    header = ""
    header += " " * index_width + "   "

    df.columns.keys.each_with_index do |c, i|
      header += "#{c}".rjust(widths[i]) + "   "
    end

    s = header + "\n"
    df.data[0].data.size.times do |i|
      s += "#{i}  ".rjust("#{df.data[0].data.size - 1}".size)
      df.data.size.times do |j|
        s += format(df.data[j].data[i].value).rjust(widths[j] + 1)
        s += "  "
      end
      s += "\n"
    end
    s[...-1]
  end
end
