function maximum_odd_binary(s)
    v = circshift(sort(collect(s) .== '1', rev=true), -1)
    String(map(b -> b ? '1' : '0', v))
end
