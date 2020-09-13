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

require "../tensor/tensor"
require "./groupby"

# A `Series` is a one dimensional `Tensor` with additional labeling
# capabilities to assist with data science and statistical modeling.
#
# A `Series` can be created from an existing `Tensor` or an enumerable,
# although all data will be copied upon creation.
class Series(T)
  getter data : Tensor(T)
  getter name : String

  # Internal initialization method for `Series`.  Some creation methods
  # will cause a data copy to be made, but if the data is coming
  # from an enumerable, data will be copied on `Tensor` creation
  # anyways, so the double copy is not necessary
  private def initialize(data : Tensor(T), name : String, copy : Bool)
    unless data.rank == 1
      raise Num::Internal::ShapeError.new("Data must be one-dimensional")
    end

    if copy
      data = data.dup(Num::RowMajor)
    end

    @data = data
    @name = name
  end

  def self.new(data : Tensor(T), name : String = "")
    new(data, name, true)
  end

  def self.new(data : Enumerable, name : String = "")
    new(data.to_tensor, name, false)
  end

  def self.new(ser : Series)
    new(ser.data, ser.name, true)
  end

  def self.new(ser : Series, name : String)
    new(ser.data, name, true)
  end

  def groupby(grouper : Tensor | Enumerable)
    SeriesGroupBy.new(self.data, grouper)
  end
end
