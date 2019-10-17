# bottle

A library for scientific computing in Crystal-Lang.  Provides memory efficient
data structures and powerful linear algebra routines backed by BLAS.  Provides
vectorized operations on one and two dimensional vectors and matrices.  Currently in
active development and not currently at all stable.  Contributions are both
welcomed and encouraged to bring powerful and fast data science to Crystal.

## Installation

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     bottle:
       github: crystal-data/bottle
   ```

2. Run `shards install`

## Usage

```crystal
require "bottle"
```

Bottle provides a Vector class as a powerful one-dimensional data structure

```crystal
v = Vector.new [1, 2, 3, 4, 5]
```

Bottle provides powerful slicing and assignment operations.

```crystal
# Sliced view of a vector
v[1...] # => [2.0, 3.0, 4.0, 5.0]

# Pass multiple indexes to return a new vector that owns its own memory
v[[0, 2, 2]] # => [1.0, 3.0, 3.0]

# Assign multiple values at once.
v[[0, 1, 2]] = [8.0, 7.0, 5.0]
v # => [8.0, 7.0, 5.0, 4.0, 5.0]
```

Bottle also provides vectorized numerical methods.

```crystal
# Cumulative operations
v.cumsum # => [1.0, 3.0, 6.0, 10.0, 15.0]

# Elementwise operations
v * v # => [1.0, 4.0, 9.0, 16.0, 25.0]

# Vectorized operations with scalars
v + 2 # => [3.0, 4.0, 5.0, 6.0, 7.0]
```

Bottle provides a 2D Matrix class for higher dimensional data.

```crystal
m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

Use linear algebra routines backed by BLAS and LAPACK

```crystal
m.matmul(m) # =>

# [[30.0, 36.0, 42.0]
#  [66.0, 81.0, 96.0]
#  [102.0, 126.0, 150.0]]
```

Apply calculations along axes of a matrix

```crystal
m.cumsum(1)

# [[1.0, 3.0, 6.0]
#  [4.0, 9.0, 15.0]
#  [7.0, 15.0, 24.0]]
```

## Development

This library is currently in very active development and I would love any help I can get!

## Contributing

1. Fork it (<https://github.com/your-github-user/bottle/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Zimmerman](https://github.com/christopherzimmerman) - creator and maintainer
