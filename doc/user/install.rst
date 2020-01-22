*****************
Installing Num.cr
*****************

Num.cr can be included in your project by adding the repository to your
`shard.yml` file.

.. code-block:: crystal

    development_dependencies:
      num:
        github: crystal-data/num.cr

Num.cr relies on several C libraries to provide fast Linear Algebra and FFT
routines.  These are: BLAS, LAPACK, and FFTW.  These libraries are not required
to use basic Tensor operations, but are needed to use linear algebra routines
or fourier transform routines.

Num.cr also supports several BLAS implementations in order to provide the best
performance on a range of operating systems.  On Darwin, use the `-Daccelerate`
compilation flag to use Darwin's accelerate framework.  You can also use
OpenBlas or Cblas using the `-Dopenblas` or `-Dcblas` flags.

Please review your operating systems installation guides on how to install
your BLAS, LAPACK, and FFTW implementations.
