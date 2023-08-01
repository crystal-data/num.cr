# Copyright (c) 2021 Crystal Data Contributors
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

class Tensor(T, S)
  # Creates a `Tensor` sampled from a provided range, with a given
  # shape.
  #
  # The generic types of the `Tensor` are inferred from the endpoints
  # of the range
  #
  # ## Arguments
  #
  # * r : `Range(U, U)` - Range of values to sample between
  # * shape : `Array(Int)` - Shape of returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # t = Tensor.random(0...10, [2, 2])
  # t
  #
  # # [[8, 4],
  # #  [7, 4]]
  # ```
  def self.random(r : Range(U, U), shape : Array(Int), device = CPU) forall U
    self.new(shape, device: device) do
      Num::Rand.stdlib_generator.rand(r)
    end
  end

  # Generate random floating point values between 0 and 1
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` Shape of `Tensor` to generate
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).rand([5])
  # puts a # => [0.411575 , 0.548264 , 0.388604 , 0.0106621, 0.183558 ]
  # ```
  def self.rand(shape : Array(Int))
    self.new(shape, device: S) do
      {% if T == Float64 %}
        Num::Rand.generator.float
      {% elsif T == Float32 %}
        Num::Rand.generator.float32
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Generates a Tensor containing a beta-distribution collection
  # of values
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * a : `Float` - Shape parameter of distribution
  # * b : `Float` - Shape parameter of distribution
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).beta([5], 0.1, 0.5)
  # puts a # => [0.000463782, 0.40858    , 1.67573e-07, 0.143055, 3.08452e-08]
  # ```
  def self.beta(shape : Array(Int), a : Float, b : Float)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * df : `Float` - Degrees of freedom
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).chisq([5], 30.0)
  # puts a # => [32.2738, 27.2351, 26.0258, 22.136 , 31.9774]
  # ```
  def self.chisq(shape : Array(Int), df : Float)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * scale : `Float` - Scale of the distribution
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).exp([5])
  # puts a # => [0.697832 , 0.710307 , 1.35733  , 0.0423776, 0.209743 ]
  # ```
  def self.exp(shape : Array(Int), scale : Float = 1.0)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * df1 : `Float` - Degrees of freedom of the underlying chi-square
  #   distribution, numerator side; usually mentioned as m.
  # * df2 : `Float` - Degrees of freedom of the underlying chi-square
  #   distribution, denominator side; usually mentioned as n.
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).fsned([5], 30.0, 50.0)
  # puts a # => [1.15436 , 1.08983 , 0.971573, 1.75811 , 2.06518 ]
  # ```
  def self.fsned(shape : Array(Int), df1 : Float, df2 : Float) : Tensor(T, S)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * t_shape : `Array(Int)` - Shape of output `Tensor`
  # * shape : `Float` - shape parameter of the distribution; usually mentioned
  #   as k
  # * scale : Float - scale parameter of the distribution; usually mentioned
  #   as θ
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).gamma([5], 0.5)
  # puts a # => [0.169394 , 0.0336937, 0.921517 , 0.0210972, 0.0487926]
  # ```
  def self.gamma(t_shape : Array(Int), shape : Float, scale : Float = 1.0) : Tensor(T, S)
    self.new(t_shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * loc : `Float` - Centrality parameter, or mean of the distribution; usually
  #   mentioned as μ
  # * scale : Float - scale parameter of the distribution; usually mentioned
  #   as b
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).laplace([5], 0.5)
  # puts a # => [0.305384 , 0.601509 , 0.247952 , -3.34791 , -0.502075]
  # ```
  def self.laplace(shape : Array(Int), loc : Float = 0.0, scale : Float = 1.0) : Tensor(T, S)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * loc : `Float` - centrality parameter, or mean of the underlying normal
  #   distribution; usually mentioned as μ
  # * sigma : `Float` - scale parameter, or standard deviation of the underlying
  #   normal distribution; usually mentioned as σ
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).lognormal([5], 0.5)
  # puts a # => [1.41285 , 5.00594 , 0.766401, 1.61069 , 2.29073 ]
  # ```
  def self.lognormal(shape : Array(Int), loc : Float = 0.0, sigma : Float = 1.0) : Tensor(T, S)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of `Tensor` to create
  # * loc : `Float` - Centrality parameter
  # * sigma : `Float` - Standard deviation
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).normal([5], 0.5)
  # puts a # => 0.345609, 1.61063 , -0.26605, 0.476662, 0.828871]
  # ```
  def self.normal(shape : Array(Int), loc = 0.0, sigma = 1.0) : Tensor(T, S)
    self.new(shape, device: S) do
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
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * lam : `Float` - Separation parameter of the distribution; usually
  #   mentioned as λ
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Int64, CPU(Int64)).poisson([5])
  # puts a # => [1, 0, 1, 0, 3]
  # ```
  def self.poisson(shape : Array(Int), lam : Float = 1.0) : Tensor(T, S)
    {% if T != Int64 %}
      {% raise "Invalid dtype #{T} for poisson distribution, only Int64 supported" %}
    {% end %}
    self.new(shape, device: S) do
      Num::Rand.generator.poisson(lam)
    end
  end

  # Generate a t-student-distributed, pseudo-random Tensor
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * df : `Float` - degrees of freedom of the distribution; usually mentioned
  #   as n
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).t_student([5], 30.0)
  # puts a # => [-0.148853, -0.803994, 0.353089 , -1.25613 , -0.141144]
  # ```
  def self.t_student(shape : Array(Int), df : Float) : Tensor(T, S)
    self.new(shape, device: S) do
      {% if T == Float64 %}
        Num::Rand.generator.t(df)
      {% elsif T == Float32 %}
        Num::Rand.generator.t32(df)
      {% else %}
        {% raise "Invalid dtype #{T} for random methods" %}
      {% end %}
    end
  end

  # Draw samples from a binomial distribution.
  # Samples are drawn from a binomial distribution with specified parameters,
  # n trials and prob probability of success where n an integer >= 0 and
  # p is in the interval [0,1].
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - Shape of output `Tensor`
  # * n : `Int` - Number of trials
  # * prob : `Float` - Probability of success on a single trial
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor(Float32, CPU(Float32)).binomial([5], 50, 0.5)
  # puts a # => [23, 30, 22, 18, 28]
  # ```
  def self.binomial(shape : Array(Int), n : Int, prob : Float) : Tensor(T, S)
    self.new(shape, device: S) do
      success = T.new(0)
      n.times do
        success += Random.rand < prob ? 1 : 0
      end
      success
    end
  end

  # Draw samples from a multinomial distribution.
  # Returns a Tensor where each row contains `num_samples` samples from the multinomial distribution
  # located in the corresponding row of Tensor `input`.
  # The rows of `input` do not need to be normalized, but must sum to a positive number.
  # If `input` is a vector (1-D Tensor), returns a vector of length `num_samples`
  # If `input` is a matrix (2-D Tensor), returns a matrix where each row contains `num_samples` samples, with shape (*m* x `num_samples`).
  #
  # ## Arguments
  #
  # * input : `Tensor` - Tensor containing probabilities of different outcomes
  # * num_samples : `Int` - Number of samples to draw from the multinomial distribution
  #
  # ## Examples
  #
  # ```
  # Num::Rand.set_seed(0)
  # input = [[0.5, 0.5], [0.5, 0.5]].to_tensor
  # a = Tensor.multinomial(input, 5)
  # puts a # => [[0, 1, 1, 0, 1], [1, 0, 1, 1, 0]]

  # input2 = [0.5, 0.5, 0.5, 0.5].to_tensor
  # b = Tensor.multinomial(input, 6)
  # puts b # => [3, 2, 1, 1, 0, 2]
  # ```
  def self.multinomial(input : Tensor(T, S), num_samples : Int32)
    sum = input.sum

    if sum == 0
      raise "Sum of probabilities is 0, can't draw samples"
    end

    # Normalize 1D tensors into 2D tensors
    if input.shape.size == 1
      input = input.expand_dims(0)
    end

    # Normalize the probabilities
    probabilities = input / input.sum(axis: 1, dims: true)

    samples = [] of Array(Int32)

    probabilities.each_axis(0) do |p_row|
      sample_set = [] of Int32
      num_samples.times do
        rand_num = Num::Rand.generator.float32

        # Calculate the cumulative probabilities
        cumulative_prob = 0.0

        # default to return the last probability
        s_index = p_row.size - 1

        # Loop through the probabilities
        p_row.each_with_index do |prob, index|
          cumulative_prob += prob
          if rand_num <= cumulative_prob
            s_index = index
            break
          end
        end

        sample_set << s_index
      end

      samples << sample_set
    end

    # If the input is a vector, return a vector of size num_samples
    if input.shape[0] == 1
      samples[0].to_tensor
    else
      samples.to_tensor
    end
  end

  private def self.draw_sample(probabilities : Array(Float64))
    # Generate a random number between 0 and 1
    rand_num = Random.new.rand

    # Calculate the cumulative probabilities
    cumulative_prob = 0.0

    # Loop through the probabilities
    probabilities.each_with_index do |prob, index|
      cumulative_prob += prob
      return index if rand_num <= cumulative_prob
    end

    # If no index has been returned, return the last one
    probabilities.size - 1
  end
end
