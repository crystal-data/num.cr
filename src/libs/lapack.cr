require "./dtype"

@[Link("lapack")]
lib LibLapack
  fun dgetrf = dgetrf_(m : Integer*, n : Integer*, a : Double*, lda : Integer*, ipiv : Integer*, info : Integer*) : Integer
  fun sgetrf = sgetrf_(m : Integer*, n : Integer*, a : Real*, lda : Integer*, ipiv : Integer*, info : Integer*) : Integer
  fun dgetri = dgetri_(n : Integer*, a : Double*, lda : Integer*, ipiv : Integer*,
                       work : Double*, lwork : Integer*, info : Integer*) : Integer
  fun sgetri = sgetri_(n : Integer*, a : Real*, lda : Integer*, ipiv : Integer*,
                       work : Real*, lwork : Integer*, info : Integer*) : Integer
end
