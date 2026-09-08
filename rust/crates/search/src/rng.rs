use catanatron_core::RandomSource;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub enum StreamKind {
    Chance = 0x4348_414e_4345,
    Policy = 0x504f_4c49_4359,
}

pub fn derive_seed(master: u64, game: u64, rollout: u64, kind: StreamKind) -> [u8; 32] {
    let mut state = master ^ game.rotate_left(17) ^ rollout.rotate_left(39) ^ kind as u64;
    let mut seed = [0_u8; 32];
    for chunk in seed.chunks_exact_mut(8) {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    seed
}

pub struct SearchRng(ChaCha8Rng);

impl SearchRng {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self(ChaCha8Rng::from_seed(seed))
    }
}

impl RandomSource for SearchRng {
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_seeds_are_stable_and_stream_separated() {
        assert_eq!(
            derive_seed(7, 11, 13, StreamKind::Chance),
            [
                74, 208, 20, 171, 254, 194, 145, 228, 189, 167, 80, 246, 220, 9, 68, 10, 11, 210,
                28, 39, 237, 66, 15, 236, 214, 105, 154, 225, 248, 222, 56, 193,
            ]
        );
        assert_ne!(
            derive_seed(7, 11, 13, StreamKind::Chance),
            derive_seed(7, 11, 13, StreamKind::Policy)
        );
    }
}
