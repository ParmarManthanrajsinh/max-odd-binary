auto maximumOddBinary(char[] s)
{
    import std.algorithm : sort, bringToFront;

    sort!"a > b"(s);
    bringToFront(s[0 .. 1], s[1 .. $]);
    return s;
}
