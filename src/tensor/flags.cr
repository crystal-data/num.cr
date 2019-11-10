@[Flags]
enum Bottle::Internal::TensorFlags
  # Contiguous really means C-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The C-style
  # part means that the data is laid out such that the last index
  # is the fastest varying as one scans though the array's
  # memory.
  Contiguous
  # Fortran really means Fortran-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The Fortran-style
  # part means that the data is laid out such that the first index
  # is the fastest varying as one scans though the array's
  # memory.
  Fortran
  # OwnData indicates if this array is the owner of the data
  # pointed to by its .ptr property.  If not then this is a
  # view onto some other array's data.
  OwnData
end
