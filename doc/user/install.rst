*****************
Installing Bottle
*****************

Bottle can be included in your project by adding the repository to your
`shard.yml` file.

.. code-block:: crystal

    development_dependencies:
      bottle:
        github: crystal-data/bottle

Bottle relies on LaPACK and BLAS under the hood for many linear algebra routines.
In order to use the library, these must be present on your machine to be properly
linked.

For Debian, use ``libopenblas-dev`` and ``liblapack-dev``. For other operating systems review the relevant
installation instructions for that OS.
