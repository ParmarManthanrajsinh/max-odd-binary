fn maximum_odd_binary(s: &[u8]) -> Vec<u8> {
    let n = s.iter().filter(|&&c| c == b'1').count();
    let mut r = Vec::with_capacity(s.len());
    r.extend(std::iter::repeat(b'1').take(n - 1));
    r.extend(std::iter::repeat(b'0').take(s.len() - n));
    r.push(b'1');
    r
}
