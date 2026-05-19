/// A bit-reversal permutation iterator for a range of size n.
/// Example: For n = 8, the iterator will yield the indices in the order:
/// 0, 4, 2, 6, 1, 5, 3, 7
/// <https://en.wikipedia.org/wiki/Bit-reversal_permutation>
pub struct BitReversalIterator {
    n: usize,     // Range size (0 to n-1)
    k: u32,       // Number of bits, smallest k where 2^k >= n
    i: usize,     // Current index
    count: usize, // Number of elements yielded
}

impl BitReversalIterator {
    #[must_use]
    pub fn new(n: usize) -> Self {
        if n == 0 {
            Self {
                n: 0,
                k: 0,
                i: 0,
                count: 0,
            }
        } else {
            // Smallest k such that 2^k >= n
            let k = 64 - n.leading_zeros();
            Self {
                n,
                k,
                i: 0,
                count: 0,
            }
        }
    }
}

impl Iterator for BitReversalIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.count < self.n {
            // Reverse bits and shift to get k least significant bits
            let rev = self.i.reverse_bits() >> (64 - self.k);
            self.i += 1;
            if rev < self.n {
                self.count += 1;
                return Some(rev);
            }
        }
        None
    }
}
