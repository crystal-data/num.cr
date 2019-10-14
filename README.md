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

Bottle provides a Vector class that supports integer and float data types.

```crystal
dv = Vector.new [1.0, 2, 3, 4, 5] # dtype is Float64
iv = Vector.new [1, 2, 3, 4, 5] # dtype is Int32

iv[1...]                  # slice of vector
iv[[1, 2, 3]]             # copy of vector, multi-indexing
iv[[1, 2, 3]] = [6, 7, 8] # in place multiple assignment
iv + iv                   # elementwise operations on vectors
iv * iv
iv / iv
iv / 5                    # elementwise operations using constants
iv - 8

iv.dot(iv)                # BLAS backed routines
iv.norm
```

## Development

TODO: Write development instructions here

## Contributing

1. Fork it (<https://github.com/your-github-user/bottle/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [Chris Zimmerman](https://github.com/christopherzimmerman) - creator and maintainer
