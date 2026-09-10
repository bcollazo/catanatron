//! Shared Python-fixture import and conformance support for the benchmark tools.

#![forbid(unsafe_code)]

mod fixture;

pub use fixture::{check_record, import_state, ConformanceError, FixtureRecord, ImportError};
