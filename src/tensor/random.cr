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

class Tensor(T)
  # Creates a `Tensor` sampled from a provided range, with a given
  # shape.
  #
  # The generic types of the `Tensor` are inferred from the endpoints
  # of the range
  #
  # Arguments
  # ---------
  # *r* : Range(U, U)
  #   Range of values to sample between
  # *shape* : Array(Int)
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # Num::Rand.set_seed(0)
  # t = Tensor.random(0...10, [2, 2])
  # t
  #
  # # [[8, 4],
  # #  [7, 4]]
  # ```
  def self.random(r : Range(U, U), shape : Array(Int)) : Tensor(U) forall U
    self.new(shape) do
      Num::Rand.stdlib_generator.rand(r)
    end
  end

  # Generate random floating point values between 0 and 1
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of Array to generate
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.rand(shape : Array(Int)) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.float
      {% elsif T == Float32 %}
        Num::Rand.generator.float32
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  private def self.binom_internal(n, prob)
    success = 0
    n.times do
      success += Random.rand < prob ? 1 : 0
    end
    success
  end

  def self.binomial(shape : Array(Int), n : Int, prob : Float) : Tensor(Int32)
    Tensor(Int32).new(shape) do
      binom_internal(n, prob)
    end
  end

  # Generates a Tensor containing a beta-distribution collection
  # of values
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # a : Float
  #   Shape parameter of distribution
  # b : Float
  #   Shape parameter of distribution
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.beta(shape : Array(Int), a : Float, b : Float) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.beta(a: a, b: b)
      {% elsif T == Float32 %}
        Num::Rand.generator.beta32(a: a, b: b)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generates a Tensor containing chi-square distributed values
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # df : Float
  #   Degrees of freedom
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.chisq(shape : Array(Int), df : Float) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.chisq(df)
      {% elsif T == Float32 %}
        Num::Rand.generator.chisq32(df)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generates a Tensor containing expontentially distributed
  # values
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # scale : Float = 1.0
  #   Scale parameter of the distribution
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.exp(shape : Array(Int), scale : Float = 1.0) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.exp(scale)
      {% elsif T == Float32 %}
        Num::Rand.generator.exp32(scale)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generates a Tensor containing f-snedecor distributed
  # values
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # df1 : Float
  #   degrees of freedom of the underlying chi-square distribution,
  #   numerator side; usually mentioned as m.
  # df2 : Float
  #   degrees of freedom of the underlying chi-square distribution,
  #   denominator side; usually mentioned as n.
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.fsned(shape : Array(Int), df1 : Float, df2 : Float) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.f(df1, df2)
      {% elsif T == Float32 %}
        Num::Rand.generator.f32(df1, df2)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generate a gamma-distributed, pseudo-random Tensor
  #
  # Arguments
  # ---------
  # t_shape : Array(Int)
  #   Shape of output Tensor
  # shape : Float
  #   shape parameter of the distribution; usually mentioned as k
  # scale : Float = 1.0
  #   scale parameter of the distribution; usually mentioned as θ
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.gamma(t_shape : Array(Int), shape : Float, scale : Float = 1.0) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.gamma(shape, scale)
      {% elsif T == Float32 %}
        Num::Rand.generator.gamma32(shape, scale)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generate a laplace-distributed, pseudo-random Tensor
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # loc : Float = 0.0
  #   centrality parameter, or mean of the distribution; usually
  #   mentioned as μ
  # scale : Float = 1.0
  #   scale parameter of the distribution; usually mentioned as b
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.laplace(shape : Array(Int), loc : Float = 0.0, scale : Float = 1.0) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.laplace(loc, scale)
      {% elsif T == Float32 %}
        Num::Rand.generator.laplace32(loc, scale)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generate a log-normal-distributed, pseudo-random Tensor
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # loc : Float = 0.0
  #   centrality parameter, or mean of the underlying normal distribution;
  #   usually mentioned as μ
  # sigma : Float = 1.0
  #   scale parameter, or standard deviation of the underlying normal
  #   distribution; usually mentioned as σ
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.lognormal(shape : Array(Int), loc : Float = 0.0, sigma : Float = 1.0) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.lognormal(loc, sigma)
      {% elsif T == Float32 %}
        Num::Rand.generator.lognormal32(loc, sigma)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generate a Tensor containing a normal-distribution collection
  # of values
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of Tensor to create
  # loc = 0.0
  #   Centrality parameter
  # sigma = 1.0
  #   Standard deviation
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.normal(shape : Array(Int), loc = 0.0, sigma = 1.0) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.normal(loc, sigma)
      {% elsif T == Float32 %}
        Num::Rand.generator.normal32(loc, sigma)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generate a poisson-distributed, pseudo-random Tensor(Int64)
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # lam : Float = 1.0
  #   separation parameter of the distribution; usually mentioned as λ
  #
  # Returns
  # -------
  # Tensor(Int64)
  #
  # Examples
  # --------
  def self.poisson(shape : Array(Int), lam : Float = 1.0) : Tensor(Int64)
    self.new(shape) do
      Num::Rand.generator.poisson(lam)
    end
  end

  # Generate a t-student-distributed, pseudo-random Tensor
  #
  # Arguments
  # ---------
  # shape : Array(Int)
  #   Shape of output Tensor
  # df : Float
  #   degrees of freedom of the distribution; usually mentioned as n
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def self.t_student(shape : Array(Int), df : Float) : Tensor(T)
    self.new(shape) do
      {% if T == Float64 %}
        Num::Rand.generator.t(df)
      {% elsif T == Float32 %}
        Num::Rand.generator.t32(df)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end
end
