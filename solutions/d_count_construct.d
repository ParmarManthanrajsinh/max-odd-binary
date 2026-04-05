auto maximumOddBinary(char[] s)
{
    import std.algorithm : count;

    auto n = s.count('1') - 1;
    s[0 .. n].fill('1');
    s[n .. $ - 1].fill('0');
    s[$ - 1] = '1';
    return s.dup;
}
