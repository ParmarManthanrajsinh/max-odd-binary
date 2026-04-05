auto maximumOddBinary(char[] s)
{
    import std.algorithm : partition, bringToFront;

    partition!(c => c == '1')(s);
    bringToFront(s[0 .. 1], s[1 .. $]);
    return s;
}
