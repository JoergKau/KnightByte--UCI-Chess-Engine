//! Network file loader and immutable weight storage.
//!
//! Two format versions are accepted:
//!
//! * **version 5** is the shipped MCNN v5 layout: one output bucket, and the
//!   historical head chain with a per-element `>> 8` before the dot product.
//! * **version 6** is the research layout used for candidate nets: one or more
//!   output buckets selected by piece count, and an exact head that squares the
//!   accumulator in full precision before multiplying by the head weight.
//!
//! A version-6 file adds a `bucket_function` field to the header (112 bytes
//! instead of 108) and stores one head row per bucket. The bucket function id is
//! written by the exporter and validated here, so the trainer, the exporter and
//! the engine cannot silently disagree about how a position maps to a bucket.
//!
//! The loader validates every architectural field, pins the quantization
//! constants to this build, checks the description bounds, verifies the payload
//! length and finally verifies the XOR checksum. A bad file is a loud refusal;
//! there is no silent fallback.

use std::fmt;

/// Network-file magic: `MCNN`.
pub const MAGIC: u32 = 0x4E4E_434D;
/// Legacy format version: one output bucket, historical head chain.
pub const VERSION_V5: u32 = 5;
/// Research format version: output buckets and the exact head.
pub const VERSION_V6: u32 = 6;
/// Supported feature scheme.
pub const FEATURE_SCHEME: i32 = 2;
/// Number of feature slots per king region.
pub const SLOTS_PER_REGION: usize = 768;
/// Accumulator clamp.
pub const QA: i32 = 255;
/// SCReLU pre-scale shift.
pub const SCRELU_SHIFT: i32 = 8;
/// Post-SCReLU shift.
pub const OUTPUT_SHIFT: i32 = 0;
/// Output-layer clamp.
pub const OUTPUT_CAP: i32 = 255;
/// Centipawn scale multiplier.
pub const EVAL_SCALE_MUL: i64 = 1600;
/// Centipawn scale shift.
pub const EVAL_SCALE_SHIFT: i64 = 20;
/// Minimum accepted hidden width.
pub const MIN_L1: usize = 64;
/// Maximum accepted hidden width.
pub const MAX_L1: usize = 2048;
/// Maximum accepted description length.
pub const MAX_DESCRIPTION: usize = 4096;
/// Maximum accepted output-bucket count.
pub const MAX_BUCKETS: usize = 32;
/// Bucket-function id for `(pieces - 2) / ceil(32 / buckets)`.
pub const BUCKET_FUNCTION_MATERIAL: i32 = 0;

/// A malformed or unsupported network file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetError(String);

impl NetError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for NetError {}

/// A fully validated network.
pub struct Net {
    region_table: [u8; 64],
    region_count: usize,
    l1: usize,
    buckets: usize,
    bucket_function: i32,
    exact_head: bool,
    description: String,
    feature_weights: Box<[i16]>,
    feature_biases: Box<[i16]>,
    output_weights: Box<[i16]>,
    output_biases: Box<[i32]>,
}

impl Net {
    /// Parses and validates a network file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NetError> {
        if bytes.len() < 112 {
            return Err(NetError::new("network file is truncated"));
        }

        let mut reader = Reader::new(bytes);
        if reader.u32()? != MAGIC {
            return Err(NetError::new("bad network magic"));
        }
        let version = reader.u32()?;
        if version != VERSION_V5 && version != VERSION_V6 {
            return Err(NetError::new("unsupported network version"));
        }
        if reader.i32()? != FEATURE_SCHEME {
            return Err(NetError::new("unsupported feature scheme"));
        }
        let declared_regions = reader.i32()?;
        if !(1..=32).contains(&declared_regions) {
            return Err(NetError::new("region count out of bounds"));
        }

        let mut region_table = [0_u8; 64];
        region_table.copy_from_slice(reader.bytes(64)?);
        let table_max = region_table.iter().copied().max().unwrap_or(0);
        let region_count = table_max as i32 + 1;
        if region_count != declared_regions {
            return Err(NetError::new(
                "region count disagrees with the region table",
            ));
        }
        if region_count.count_ones() != 1 || !(1..=32).contains(&region_count) {
            return Err(NetError::new(
                "region count must be a power of two in 1..=32",
            ));
        }
        let mut used = [false; 32];
        for region in region_table {
            used[region as usize] = true;
        }
        for (region, is_used) in used.iter().take(region_count as usize).enumerate() {
            if !is_used {
                return Err(NetError::new(format!(
                    "region table contains unused region {region}"
                )));
            }
        }

        let l1 = reader.i32()? as usize;
        if !(MIN_L1..=MAX_L1).contains(&l1) || !l1.is_multiple_of(16) {
            return Err(NetError::new("hidden width is out of bounds or misaligned"));
        }
        let buckets = reader.i32()?;
        if !(1..=MAX_BUCKETS as i32).contains(&buckets) {
            return Err(NetError::new("output bucket count out of bounds"));
        }
        let buckets = buckets as usize;
        let (bucket_function, header_length) = if version == VERSION_V6 {
            let function = reader.i32()?;
            if function != BUCKET_FUNCTION_MATERIAL {
                return Err(NetError::new("unsupported output bucket function"));
            }
            (function, 112)
        } else {
            if buckets != 1 {
                return Err(NetError::new(
                    "version 5 supports exactly one output bucket",
                ));
            }
            (BUCKET_FUNCTION_MATERIAL, 108)
        };
        if reader.i32()? != QA
            || reader.i32()? != SCRELU_SHIFT
            || reader.i32()? != OUTPUT_SHIFT
            || reader.i32()? != EVAL_SCALE_MUL as i32
            || reader.i32()? != EVAL_SCALE_SHIFT as i32
        {
            return Err(NetError::new(
                "network quantization differs from this build",
            ));
        }

        let description_length = reader.i32()?;
        if description_length < 0 || description_length as usize > MAX_DESCRIPTION {
            return Err(NetError::new("network description length is out of bounds"));
        }
        let description_bytes = reader.bytes(description_length as usize)?;
        let description = std::str::from_utf8(description_bytes)
            .map_err(|_| NetError::new("network description is not UTF-8"))?
            .to_owned();

        let feature_rows = SLOTS_PER_REGION * region_count as usize;
        let feature_weights = reader.i16s(feature_rows * l1)?;
        let feature_biases = reader.i16s(l1)?;
        let output_weights = reader.i16s(buckets * 2 * l1)?;
        let mut output_biases = Vec::with_capacity(buckets);
        for _ in 0..buckets {
            output_biases.push(reader.i32()?);
        }

        let expected_length = header_length
            + 4
            + description_length as usize
            + feature_rows * l1 * 2
            + l1 * 2
            + buckets * 2 * l1 * 2
            + buckets * 4
            + 4;
        if bytes.len() != expected_length {
            return Err(NetError::new(
                "network file has an unexpected payload length",
            ));
        }

        let checksum_offset = bytes.len() - 4;
        let stored = reader.u32_at(checksum_offset)?;
        let actual = xor_word_checksum(bytes, header_length, checksum_offset);
        if actual != stored {
            return Err(NetError::new("network checksum mismatch"));
        }

        Ok(Self {
            region_table,
            region_count: region_count as usize,
            l1,
            buckets,
            bucket_function,
            exact_head: version == VERSION_V6,
            description,
            feature_weights: feature_weights.into_boxed_slice(),
            feature_biases: feature_biases.into_boxed_slice(),
            output_weights: output_weights.into_boxed_slice(),
            output_biases: output_biases.into_boxed_slice(),
        })
    }

    /// Returns the raw king-region table.
    #[inline(always)]
    pub fn region_table(&self) -> &[u8; 64] {
        &self.region_table
    }

    /// Returns the number of king regions.
    #[inline(always)]
    pub fn region_count(&self) -> usize {
        self.region_count
    }

    /// Returns the hidden width.
    #[inline(always)]
    pub fn l1(&self) -> usize {
        self.l1
    }

    /// Returns the number of output buckets.
    #[inline(always)]
    pub fn bucket_count(&self) -> usize {
        self.buckets
    }

    /// Returns the output-bucket function id stored in the header.
    #[inline(always)]
    pub fn bucket_function(&self) -> i32 {
        self.bucket_function
    }

    /// Returns `true` when the head uses the exact (full-precision) square.
    #[inline(always)]
    pub fn exact_head(&self) -> bool {
        self.exact_head
    }

    /// Maps a piece count to its output bucket.
    ///
    /// A legal position has at least the two kings; the clamp only keeps the
    /// helper total for a caller that passes an impossible count.
    #[inline(always)]
    pub fn bucket_for_pieces(&self, pieces: u32) -> usize {
        let divisor = 32_usize.div_ceil(self.buckets);
        ((pieces.max(2) as usize - 2) / divisor).min(self.buckets - 1)
    }

    /// Returns the network description from the file header.
    pub fn description(&self) -> &str {
        &self.description
    }

    pub(crate) fn feature_weights(&self) -> &[i16] {
        &self.feature_weights
    }

    pub(crate) fn feature_biases(&self) -> &[i16] {
        &self.feature_biases
    }

    /// Returns the `2 * L1` head weights of one bucket.
    pub(crate) fn output_weights(&self, bucket: usize) -> &[i16] {
        debug_assert!(bucket < self.buckets);
        let row = 2 * self.l1;
        &self.output_weights[bucket * row..(bucket + 1) * row]
    }

    /// Returns the head bias of one bucket.
    pub(crate) fn output_bias(&self, bucket: usize) -> i32 {
        debug_assert!(bucket < self.buckets);
        self.output_biases[bucket]
    }
}

fn xor_word_checksum(bytes: &[u8], start: usize, end: usize) -> u32 {
    let mut checksum = 0_u32;
    let mut offset = start;
    while offset + 4 <= end {
        checksum ^= u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        offset += 4;
    }
    checksum
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], NetError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| NetError::new("network offset overflow"))?;
        if end > self.bytes.len() {
            return Err(NetError::new("network file is truncated"));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, NetError> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u32_at(&self, offset: usize) -> Result<u32, NetError> {
        if offset + 4 > self.bytes.len() {
            return Err(NetError::new("network file is truncated"));
        }
        Ok(u32::from_le_bytes([
            self.bytes[offset],
            self.bytes[offset + 1],
            self.bytes[offset + 2],
            self.bytes[offset + 3],
        ]))
    }

    fn i32(&mut self) -> Result<i32, NetError> {
        Ok(self.u32()? as i32)
    }

    fn i16s(&mut self, count: usize) -> Result<Vec<i16>, NetError> {
        let bytes = self.bytes(count * 2)?;
        let mut values = Vec::with_capacity(count);
        for chunk in bytes.as_chunks::<2>().0 {
            values.push(i16::from_le_bytes(*chunk));
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUCKET_FUNCTION_MATERIAL, EVAL_SCALE_MUL, EVAL_SCALE_SHIFT, MAGIC, Net, OUTPUT_SHIFT, QA,
        SCRELU_SHIFT, VERSION_V5, VERSION_V6, xor_word_checksum,
    };
    use crate::tools::net::bytes;

    /// Builds a small synthetic version-6 file for the loader tests.
    ///
    /// The payload is deterministic and its head weights count down, so a test
    /// can distinguish the bucket rows.
    fn v6_bytes(buckets: usize, l1: usize) -> Vec<u8> {
        let regions = 1_usize;
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC.to_le_bytes());
        data.extend_from_slice(&VERSION_V6.to_le_bytes());
        data.extend_from_slice(&2_i32.to_le_bytes());
        data.extend_from_slice(&(regions as i32).to_le_bytes());
        data.extend_from_slice(&[0_u8; 64]);
        data.extend_from_slice(&(l1 as i32).to_le_bytes());
        data.extend_from_slice(&(buckets as i32).to_le_bytes());
        data.extend_from_slice(&BUCKET_FUNCTION_MATERIAL.to_le_bytes());
        data.extend_from_slice(&QA.to_le_bytes());
        data.extend_from_slice(&SCRELU_SHIFT.to_le_bytes());
        data.extend_from_slice(&OUTPUT_SHIFT.to_le_bytes());
        data.extend_from_slice(&(EVAL_SCALE_MUL as i32).to_le_bytes());
        data.extend_from_slice(&(EVAL_SCALE_SHIFT as i32).to_le_bytes());
        assert_eq!(data.len(), 112);

        let description = b"test";
        data.extend_from_slice(&(description.len() as i32).to_le_bytes());
        data.extend_from_slice(description);
        for index in 0..768 * regions * l1 {
            data.extend_from_slice(&((index % 101) as i16 - 50).to_le_bytes());
        }
        for index in 0..l1 {
            data.extend_from_slice(&(index as i16).to_le_bytes());
        }
        for index in 0..buckets * 2 * l1 {
            data.extend_from_slice(&((index as i16) - 7).to_le_bytes());
        }
        for bucket in 0..buckets {
            data.extend_from_slice(&(bucket as i32 * 3 - 5).to_le_bytes());
        }
        let checksum = xor_word_checksum(&data, 112, data.len());
        data.extend_from_slice(&checksum.to_le_bytes());
        data
    }

    #[test]
    fn embedded_network_loads() {
        let net = Net::from_bytes(bytes()).expect("embedded network must load");
        assert_eq!(net.l1(), 512);
        assert_eq!(net.region_count(), 8);
        assert_eq!(net.bucket_count(), 1);
        assert!(!net.exact_head());
    }

    #[test]
    fn version_six_networks_load_with_buckets() {
        let net = Net::from_bytes(&v6_bytes(8, 64)).expect("synthetic v6 net must load");
        assert_eq!(net.l1(), 64);
        assert_eq!(net.bucket_count(), 8);
        assert_eq!(net.bucket_function(), BUCKET_FUNCTION_MATERIAL);
        assert!(net.exact_head());
        assert_eq!(net.output_weights(0)[0], -7);
        assert_eq!(net.output_weights(1)[0], 2 * 64 - 7);
        assert_eq!(net.output_bias(0), -5);
        assert_eq!(net.output_bias(7), 16);
        assert_eq!(net.bucket_for_pieces(2), 0);
        assert_eq!(net.bucket_for_pieces(32), 7);
    }

    #[test]
    fn version_six_rejects_an_unknown_bucket_function() {
        let mut data = v6_bytes(8, 64);
        data[88..92].copy_from_slice(&99_i32.to_le_bytes());
        let checksum = xor_word_checksum(&data, 112, data.len() - 4);
        let last = data.len() - 4;
        data[last..].copy_from_slice(&checksum.to_le_bytes());
        assert!(Net::from_bytes(&data).is_err());
    }

    #[test]
    fn version_five_rejects_output_buckets() {
        let mut data = bytes().to_vec();
        data[84..88].copy_from_slice(&8_i32.to_le_bytes());
        let checksum = xor_word_checksum(&data, 108, data.len() - 4);
        let last = data.len() - 4;
        data[last..].copy_from_slice(&checksum.to_le_bytes());
        assert!(Net::from_bytes(&data).is_err());
    }

    #[test]
    fn the_shipped_file_is_reported_as_version_five() {
        let data = bytes();
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, VERSION_V5);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut data = bytes().to_vec();
        data[0] ^= 0xff;
        assert!(Net::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut data = bytes().to_vec();
        data[4] = 4;
        assert!(Net::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_checksum_mismatch() {
        let mut data = bytes().to_vec();
        let last = data.len() - 1;
        data[last] ^= 0xff;
        assert!(Net::from_bytes(&data).is_err());
    }

    #[test]
    fn rejects_truncated_payload() {
        let data = bytes();
        assert!(Net::from_bytes(&data[..data.len() - 8]).is_err());
    }
}
