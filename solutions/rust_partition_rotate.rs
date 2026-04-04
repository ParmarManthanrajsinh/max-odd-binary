#![feature(iter_partition_in_place)]

fn maximum_odd_binary(mut s: Vec<u8>) -> Vec<u8> {
    s.iter_mut().partition_in_place(|c| *c == b'1');
    s.rotate_left(1);
    s
}
