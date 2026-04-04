import std.algorithm : sort, bringToFront;

auto maximumOddBinary(char[] s) {
    sort!"a > b"(s);
    bringToFront(s[0 .. 1], s[1 .. $]);
    return s;
}
