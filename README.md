![bottle](https://raw.githubusercontent.com/crystal-data/bottle/master/static/bottle_logo.png)

[![Build Status](https://travis-ci.org/crystal-data/bottle.svg?branch=master)](https://travis-ci.org/crystal-data/bottle)

Bottle is the core shard needed for scientific computing with Crystal

- **Website:** https://www.crystaldata.org
- **API Documentation:** http://crystaldata.org/bottle
- **Source code:** https://github.com/crystaldata/bottle
- **Bug reports:** https://github.com/crystaldata/bottle/issues

It provides:

- efficient 1 and 2-dimensional data structures
- sophisticated reduction and accumulation routines
- data structures that can easily be passed to C libraries
- powerful linear algebra routines backed by LAPACK and BLAS

Roadmap

- [x] Robust `Vector` implementation for one-dimensional data
- [x] `Vector` arithmetic implemented, backed by BLAS when possible
- [x] `Vector` statistics implemented, backed by BLAS when possible
- [ ] `Vector` accumulations and reductions implemented
- [ ] Robust `Matrix` implementation for two-dimensional data
- [ ] `Matrix` arithmetic implemented, backed by BLAS when possible
- [ ] `Matrix` statistics implemented, backed by BLAS when possible
- [ ] `Matrix` accumulations and reductions along axes implemented
- [ ] Boolean masks implemented for `Matrix` and `Vector`
- [ ] Higher level Linear Algebra implementations
- [ ] Reading and writing data to files

Contributing
------------
Bottle requires help in many different ways to continue to grow as a shard.
Contributions such as high level documentation and code quality checks are needed just
as much as API enhancements.  If you are considering larger scale contributions
that extend beyond minor enhancements and bug fixes, contact Crystal Data
in order to be added to the organization to gain access to review and merge
permissions.
