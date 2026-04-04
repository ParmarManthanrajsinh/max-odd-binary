import std.algorithm : partition, bringToFront;

auto maximumOddBinary(char[] s) {
    partition!(c => c == '1')(s);
    bringToFront(s[0 .. 1], s[1 .. $]);
    return s;
}
