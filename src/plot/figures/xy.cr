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

class Num::Plot::XYPlot < Num::Plot::Figure
  @x : Tensor(Float64)
  @y : Tensor(Float64)
  @size : Int32
  @color : Int32? = nil

  # Initializes a basic XY plot
  #
  # Arguments
  # ---------
  # x
  #   Tensor-like x-axis argument
  # y
  #   Tensor-like y-axis argument
  # @color : Int32? = nil
  #   Color code for the plot
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def initialize(x, y, @color : Int32? = nil)
    @x = x.to_tensor.as_type(Float64)
    @y = y.to_tensor.as_type(Float64)

    unless @x.size == @y.size
      raise Num::Internal::ShapeError.new("Inputs must be the same size")
    end

    @size = @x.size
  end

  # Base plotting method, sets color to a default value
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def plot
    unless @color.nil?
      LibPlplot.plcol0(@color.unsafe_as(Int32))
    end
  end

  def update_bounds(bounds : Num::Plot::Bounds) : Num::Plot::Bounds
    x_min, x_max = @x.min, @x.max
    y_min, y_max = @y.min, @y.max

    bounds.x_min = {bounds.x_min, x_min}.min
    bounds.x_max = {bounds.x_max, x_max}.max
    bounds.y_min = {bounds.y_min, y_min}.min
    bounds.y_max = {bounds.y_max, y_max}.max
    bounds
  end
end
