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

module Num::Plot
  # Sets the color index for cmap0 (see the section called “Color Map0”).
  #
  # 0	black (default background)
  # 1	red (default foreground)
  # 2	yellow
  # 3	green
  # 4	aquamarine
  # 5	pink
  # 6	wheat
  # 7	grey
  # 8	brown
  # 9	blue
  # 10	BlueViolet
  # 11	cyan
  # 12	turquoise
  # 13	magenta
  # 14	salmon
  # 15	white
  # Use plscmap0 to change the entire cmap0 color palette and plscol0 to
  # change an individual color in the cmap0 color palette.
  enum Color
    BLACK       =  0
    RED         =  1
    YELLOW      =  2
    GREEN       =  3
    AQUAMARINE  =  4
    PINK        =  5
    WHEAT       =  6
    GREY        =  7
    BROWN       =  8
    BLUE        =  9
    BLUE_VIOLET = 10
    CYAN        = 11
    TURQUOISE   = 12
    MAGENTA     = 13
    SALMON      = 14
    WHITE       = 15
  end
end
