import arraymancer, os, strutils, strformat


proc elementwise*(n: int): auto =
  let i = randomTensor[float]([n], 1.0)
  let j = randomTensor[float]([n], 1.0)
  let k = randomTensor[float]([n], 1.0)
  map3_inline(i, j, k):
    1.0 + x / 1.0 + y * z

proc main() =
  var n = 100

  if paramCount()>0:
    n = parseInt(paramStr(1))

  echo ""
  echo fmt"Arraymancer elementwise {n}"
  let c = elementwise(n)
  echo formatFloat(c[n div 2], ffDefault, 16)

main()
