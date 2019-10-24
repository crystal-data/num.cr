require "../blas/level_one"

module Bottle::B
  extend self

  # Return evenly spaced values within a given interval.
  #
  # Values are generated within the half-open interval [start, stop)
  # (in other words, the interval including start but excluding stop).
  #
  # ```crystal
  # B.vrange(1, 5) # => Vector[1, 2, 3, 4]
  # ```
  def vrange(start : Int32, stop : Int32, step : Int32 = 1, dtype : U.class = Int32) forall U
    r = stop - start
    num = r // step
    if stop <= start || !num
      raise "vrange must return at least one value"
    end
    Vector.new(num) { |i| U.new(start + (i * step)) }
  end

  # Return evenly spaced values within a given interval.
  #
  # Values are generated within the half-open interval [start, stop)
  # (in other words, the interval including start but excluding stop).
  #
  # ```crystal
  # B.vrange(5) # => Vector[0, 1, 2, 3, 4]
  # ```
  def vrange(stop : Int32, step : Int32 = 1, dtype : U.class = Int32) forall U
    vrange(0, stop, step, dtype)
  end

  # Return evenly spaced numbers over a specified interval.
  # Returns `num` evenly spaced samples, calculated over the
  # interval [`start`, `stop`].
  # The endpoint of the interval can optionally be excluded.
  #
  # ```crystal
  # B.linspace(0, 1, 5) # => Vector[0.0, 0.25, 0.5, 0.75, 1.0]
  #
  # B.linspace(0, 1, 5, endpoint: false) # => Vector[0.0, 0.2, 0.4, 0.6, 0.8]
  # ```
  def linspace(start : Number, stop : Number, num = 50, endpoint = true)
    if num < 0
      raise "Number of samples, #{num}, must be non-negative"
    end
    div = endpoint ? num - 1 : num
    start = start * 1.0
    stop = stop * 1.0
    y = vrange(num, dtype: Float64)
    delta = stop - start
    if num > 1
      step = delta / div
      if step == 0
        raise "Cannot have a step of 0"
      end
      y = y * step
    else
      y = y * delta
    end
    y += start
    if endpoint && num > 1
      y[y.size - 1] = stop
    end
    y
  end

  # Return numbers spaced evenly on a log scale.
  # In linear space, the sequence starts at ``base ** start``
  # (`base` to the power of `start`) and ends with ``base ** stop``
  # (see `endpoint` below).
  #
  # ```crystal
  # B.logspace(2.0, 3.0, num = 4) # => Vector[100.0, 215.44346900318845, 464.15888336127773, 1000.0]
  # ```
  def logspace(start, stop, num = 50, endpoint = true, base = 10.0)
    y = linspace(start, stop, num: num, endpoint: endpoint)
    return power(base, y)
  end

  # Return numbers spaced evenly on a log scale (a geometric progression).
  # This is similar to `logspace`, but with endpoints specified directly.
  # Each output sample is a constant multiple of the previous.
  #
  # ```
  # geomspace(1, 1000, 4) # => Vector[1.0, 10.0, 100.0, 1000.0]
  # ```
  def geomspace(start, stop, num = 50, endpoint = true)
    if start == 0 || stop == 0
      raise "Geometric sequence cannot include zero"
    end

    out_sign = 1.0

    if start < 0 && stop < 0
      start, stop = -start, -stop
      out_sign = -out_sign
    end

    log_start = Math.log(start, 10.0)
    log_stop = Math.log(stop, 10.0)

    logspace(log_start, log_stop, num: num, endpoint: endpoint, base: 10.0) * out_sign
  end
end
