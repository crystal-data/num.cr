#include <stdio.h>
#include <gsl/gsl_vector.h>

const int MULTIPLICITY = 1;

gsl_vector_int *
gsl_vector_ma_equal (gsl_vector * u, gsl_vector * v)
{
  const size_t n = v->size;
  const size_t stride_u = u->stride;
  const size_t stride_v = v->stride;

  gsl_vector_int * out = gsl_vector_int_calloc(n);

  size_t j;

  if (u->size != v->size)
    {
      GSL_ERROR_VAL ("vectors must have same length", GSL_EBADLEN, 0);
    }

  for (j = 0; j < n; j++)
    {
      size_t k;

      for (k = 0; k < MULTIPLICITY; k++)
        {
          if (u->data[MULTIPLICITY * stride_u * j + k] == v->data[MULTIPLICITY * stride_v * j + k])
            {
              out->data[j] = 1;
            }
        }
    }

  return out;
}
