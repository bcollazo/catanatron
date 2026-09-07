//! Checked dense identifiers for fixed board tables.
use core::fmt;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdError {
    OutOfRange {
        kind: &'static str,
        value: u8,
        max: u8,
    },
}
impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { kind, value, max } => write!(f, "{kind} {value} exceeds {max}"),
        }
    }
}
macro_rules! id {
    ($name:ident, $max:expr, $label:literal) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u8);
        impl $name {
            pub const MAX: u8 = $max;
            pub fn new(value: u8) -> Result<Self, IdError> {
                if value <= Self::MAX {
                    Ok(Self(value))
                } else {
                    Err(IdError::OutOfRange {
                        kind: $label,
                        value,
                        max: Self::MAX,
                    })
                }
            }
            #[must_use]
            pub const fn get(self) -> u8 {
                self.0
            }
        }
    };
}
id!(PlayerId, 3, "player id");
id!(NodeId, 53, "node id");
id!(EdgeId, 71, "edge id");
id!(TileId, 18, "tile id");
