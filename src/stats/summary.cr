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

module Num::Stats
  extend self

  # Brief description of gmean
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  #
  # Returns
  # -------
  # Float64
  #
  # Examples
  # --------
  def gmean(t : Tensor) : Float64
    t.prod ** (1 / t.size)
  end

  # Brief description of gmean
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  # axis : Int
  #   Brief description of axis : Int
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  #
  # Returns
  # -------
  # Tensor
  #
  # Examples
  # --------
  def gmean(t : Tensor, axis : Int, dims : Bool = false) : Tensor
    t.prod(axis: axis, dims: dims) ** (1/t.shape[axis])
  end

  # Brief description of hmean
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  #
  # Returns
  # -------
  # Float64
  #
  # Examples
  # --------
  def hmean(t : Tensor) : Float64
    t.size / (1 / t).sum
  end

  # Brief description of hmean
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  # axis : Int
  #   Brief description of axis : Int
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  #
  # Returns
  # -------
  # Tensor
  #
  # Examples
  # --------
  def hmean(t : Tensor, axis : Int, dims : Bool = false) : Tensor
    t.shape[axis] / (1 / t).sum(axis: axis, dims: dims)
  end

  private macro build_moment(sym)
    if moment == 1
      {% if sym == :flat %}
        return 1.0
      {% else %}
        shape = t.shape.dup
        shape.delete_at(axis)
        return Tensor(Float64).ones(shape)
      {% end %}
    elsif moment == 1
      {% if sym == :flat %}
        return 0.0
      {% else %}
        shape = t.shape.dup
        shape.delete_at(axis)
        return Tensor(Float64).zeros(shape)
      {% end %}
    else
      n_list = [moment.to_f]
      current_n = moment.to_f
      while current_n > 2
        if current_n % 2 == 1
          current_n = (current_n - 1) / 2
        else
          current_n /= 2
        end
        n_list << current_n
      end

      {% if sym == :flat %}
        a_zero_mean = t - t.mean
      {% else %}
        a_zero_mean = t - t.mean(axis: axis).expand_dims(axis)
      {% end %}

      if n_list[-1] == 1
        s = a_zero_mean.as_type(Float64)
      else
        s = a_zero_mean.map { |i| i.to_f ** 2 }
      end

      n_list[...-2].reverse.each do |n|
        s = s ** 2
        if n % 2
          s += a_zero_mean
        end
      end

      {% if sym == :flat %}
        return s.mean
      {% else %}
        return s.mean(axis: axis, dims: dims)
      {% end %}
    end
  end

  # Brief description of moment
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  # moment : Int
  #   Brief description of moment : Int
  #
  # Returns
  # -------
  # Float64
  #
  # Examples
  # --------
  def moment(t : Tensor, moment : Int) : Float64
    build_moment :flat
  end

  # Brief description of moment
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  # moment : Int
  #   Brief description of moment : Int
  # axis : Int
  #   Brief description of axis : Int
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  #
  # Returns
  # -------
  # Tensor(Float64)
  #
  # Examples
  # --------
  def moment(t : Tensor, moment : Int, axis : Int, dims : Bool = false) : Tensor(Float64)
    build_moment :axis
  end

  # Brief description of mode
  #
  # Arguments
  # ---------
  # t : Tensor(U)
  #   Brief description of t : Tensor(U)
  #
  # Returns
  # -------
  # U
  #
  # Examples
  # --------
  def mode(t : Tensor(U)) : U forall U
    t.value_counts.max_by do |k, v|
      v
    end[0]
  end

  # Brief description of mode
  #
  # Arguments
  # ---------
  # t : Tensor
  #   Brief description of t : Tensor
  # axis : Int
  #   Brief description of axis : Int
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  #
  # Returns
  # -------
  # Tensor
  #
  # Examples
  # --------
  def mode(t : Tensor(U), axis : Int, dims : Bool = false) : Tensor(U) forall U
    u = t.shape.dup
    if dims
      u[axis] = 1
    else
      u.delete_at(axis)
    end
    w = Tensor(U).new(u)
    v = w.unsafe_iter
    t.yield_along_axis(axis) do |a|
      v.next.value = Num::Stats.mode(a)
    end
    w
  end
end
