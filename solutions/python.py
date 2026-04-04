import numpy as np


def maximum_odd_binary(s):
    s = list(s)
    s.sort(reverse=True)
    s.append(s.pop(0))
    return "".join(s)


def maximum_odd_binary_numpy(s):
    s = np.array(list(s))
    s = np.sort(s)[::-1]
    s = np.append(s[1:], s[0])
    return "".join(s)


print(maximum_odd_binary("1011"))
print(maximum_odd_binary("100"))
print(maximum_odd_binary("111000"))
print(maximum_odd_binary("0101"))
print(maximum_odd_binary("1111"))

print(maximum_odd_binary_numpy("1011"))
print(maximum_odd_binary_numpy("100"))
print(maximum_odd_binary_numpy("111000"))
print(maximum_odd_binary_numpy("0101"))
print(maximum_odd_binary_numpy("1111"))
