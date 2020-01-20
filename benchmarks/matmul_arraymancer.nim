import arraymancer, os, strutils, strformat


proc matmul*(n: int): auto =
  let a = randomTensor[float]([n, n], 1.0)
  let b = randomTensor[float]([n, n], 1.0)
  a * b

proc main() =
  var n = 100

  if paramCount()>0:
    n = parseInt(paramStr(1))

  echo ""
  echo fmt"Arraymancer matmul {n}x{n}"
  let c = matmul(n)
  echo formatFloat(c[n div 2, n div 2], ffDefault, 16)

main()
