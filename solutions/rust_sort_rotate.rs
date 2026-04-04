fn maximum_odd_binary(mut s: Vec<u8>) -> Vec<u8> {
    s.sort_unstable_by(|a, b| b.cmp(a));
    s.rotate_left(1);
    s
}
