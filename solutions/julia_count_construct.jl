function maximum_odd_binary(s)
    n = count(==('1'), s)
    '1'^(n-1) * '0'^(length(s)-n) * "1"
end
