require "../tensor/tensor"

module Num::Einsum

  def preprocess(x : Tensor, axes : Array(Int32), flip : Bool)
    axes = axes.map do |i|
      i >= 0 ? i : i + x.ndims
    end

    free = [] of Int32
    (0...x.ndims).each do |i|
      if !axes.includes?(i)
        free << i
      end
    end

    free_dims = free.map do |i|
      x.shape[i]
    end

    prod_free = free_dims.product
    prod_axes = axes.map {|a| x.shape[a]}.product

    perm = flip ? axes + free : free + axes
    new_shape = flip ? [prod_axes, prod_free] : [prod_free, prod_axes]

    reshaped = x.transpose(perm).reshape(new_shape).dup
    {reshaped, free_dims}
  end

  def tensordot(a, b, axes : Tuple(Array(Int32), Array(Int32)))
    axes_a, axes_b = axes
    na = axes_a.size
    nb = axes_b.size

    as_ = a.shape
    nda = a.ndims
    bs = b.shape
    ndb = b.ndims
    equal = true
    if na != nb
      equal = false
    else
      na.times do |k|
        if as_[axes_a[k]] != bs[axes_b[k]]
          equal = false
          break
        end
        if axes_a[k] < 0
          axes_a[k] += nda
        end
        if axes_b[k] < 0
          axes_b[k] += ndb
        end
      end
    end

    if !equal
      raise "shape-mismatch for sum"
    end

    notin = (0...nda).to_a
    notin.select! { |e| !axes_a.includes?(e) }
    newaxes_a = notin + axes_a
    n2 = 1
    axes_a.each do |axis|
      n2 *= as_[axis]
    end

    new_shape_a = [notin.map { |ax| as_[ax] }.product, n2]
    olda = notin.map { |ax| as_[ax]  }

    notin = (0...ndb).to_a
    notin.select! { |e| !axes_b.includes?(e) }
    newaxes_b = axes_b + notin
    n2 = 1
    axes_b.each do |axis|
      n2 *= bs[axis]
    end

    new_shape_b = [n2, notin.map { |ax| bs[ax] }.product]
    oldb = notin.map { |ax| bs[ax]  }

    at = a.transpose(newaxes_a).reshape(new_shape_a)
    bt = b.transpose(newaxes_b).reshape(new_shape_b)

    at.matmul(bt).reshape(olda + oldb)
  end
end
