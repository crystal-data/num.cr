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

require "../figures/*"

class Num::Plot::Options
  getter bounds : Num::Plot::Bounds = Num::Plot::Bounds.new
  getter figures : Array(Num::Plot::Figure) = [] of Num::Plot::Figure

  private macro tap_prop(name, dtype, default)
    @{{name}} : {{dtype}} = {{ default }}

    def {{name}}
      @{{name}}
    end

    def {{name}}(val : {{dtype}})
      @{{name}} = val
    end
  end

  tap_prop term, Symbol?, :qtwidget
  tap_prop palette, Symbol, :alternate
  tap_prop x_label, String, ""
  tap_prop y_label, String, ""
  tap_prop label, String, ""

  def initialize
  end

  # Plots a line plot
  #
  # Arguments
  # ---------
  # x
  #   Tensor-like x-axis data
  # y
  #   Tensor-like y-axis data
  # color = nil
  #   Color code for data
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  # ```
  # Num::Plot::Plot.plot do
  #   line [1, 2, 3], [4, 5, 3]
  # end
  # ```
  def line(x, y, color = nil)
    plt = Num::Plot::LinePlot.new(x, y, color)
    @figures << plt
    @bounds = plt.update_bounds(@bounds)
  end

  # Plots a scatter plot
  #
  # Arguments
  # ---------
  # x
  #   Tensor like x-axis data
  # y
  #   Tensor-like y-axis data
  # color = nil
  #   Color code for plot
  # code : Int32 = 1
  #   Symbol to use for points
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  # ```
  # Num::Plot::Plot.plot do
  #   scatter [3, 4, 2], [1, 7, 8], 3, 17
  # end
  # ```
  def scatter(x, y, color = nil, code : Int32 = 1)
    plt = Num::Plot::Scatter.new(x, y, color, code)
    @figures << plt
    @bounds = plt.update_bounds(@bounds)
  end

  # Plots a custom plot.  As long as a user defines a class
  # that inherits from Figure, this method will ensure it is
  # added to the Plot.  This allows users to add custom plots
  # using the block syntax
  #
  # Arguments
  # ---------
  # cls : U.class
  #   Class to use for plotting
  # *args
  #   Arguments to create a plot
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def custom(cls : U.class, *args) forall U
    plt = U.new(*args)
    @figures << plt
    @bounds = plt.update_bounds(@bounds)
  end
end
