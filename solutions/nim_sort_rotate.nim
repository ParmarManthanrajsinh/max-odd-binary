import std/algorithm

proc maximumOddBinary(s: var string) =
  sort(s, order = Descending)
  rotateLeft(s, 1)
