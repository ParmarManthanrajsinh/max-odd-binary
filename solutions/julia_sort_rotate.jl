function maximum_odd_binary(s)
    v = sort(s, rev=true)
    circshift(v, -1)
end
