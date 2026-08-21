//! Pure host-side parsing/averaging of Bullet's serialized weight format.
//! The format is `{ascii id}\n{usize LE len}{len * f32 LE}`, with tensors
//! concatenated in sorted-id order.

use std::{collections::BTreeMap, io::{self, ErrorKind}};

const MAX_TENSOR_ID_BYTES: usize = 256;

pub fn parse(bytes: &[u8]) -> io::Result<BTreeMap<String, Vec<f32>>> {
    let mut map = BTreeMap::new();
    let mut offset = 0;

    while offset < bytes.len() {
        let id_end = bytes[offset..]
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| invalid_data("checkpoint tensor id is missing its newline"))?;
        if id_end == 0 || id_end > MAX_TENSOR_ID_BYTES {
            return Err(invalid_data("checkpoint tensor id has an invalid length"));
        }
        let id_bytes = &bytes[offset..offset + id_end];
        if !id_bytes.is_ascii() {
            return Err(invalid_data("checkpoint tensor id is not ASCII"));
        }
        let id = String::from_utf8(id_bytes.to_vec())
            .map_err(|_| invalid_data("checkpoint tensor id is not UTF-8"))?;
        offset += id_end + 1;

        let word_bytes = size_of::<usize>();
        let len_end = offset.checked_add(word_bytes).ok_or_else(|| invalid_data("checkpoint length overflow"))?;
        if len_end > bytes.len() {
            return Err(invalid_data("checkpoint tensor length is truncated"));
        }
        let len = usize::from_le_bytes(bytes[offset..len_end].try_into().expect("fixed-size slice"));
        offset = len_end;
        let value_bytes = len.checked_mul(size_of::<f32>()).ok_or_else(|| invalid_data("checkpoint tensor length overflows"))?;
        let values_end = offset.checked_add(value_bytes).ok_or_else(|| invalid_data("checkpoint tensor payload overflows"))?;
        if values_end > bytes.len() {
            return Err(invalid_data("checkpoint tensor payload is truncated"));
        }

        let mut values = Vec::with_capacity(len);
        for value in bytes[offset..values_end].chunks_exact(size_of::<f32>()) {
            let value = f32::from_le_bytes(value.try_into().expect("fixed-size slice"));
            if !value.is_finite() {
                return Err(invalid_data("checkpoint contains a non-finite weight"));
            }
            values.push(value);
        }
        offset = values_end;
        if map.insert(id.clone(), values).is_some() {
            return Err(invalid_data(&format!("checkpoint contains duplicate tensor `{id}`")));
        }
    }

    if map.is_empty() {
        return Err(invalid_data("checkpoint is empty"));
    }
    Ok(map)
}

pub fn serialize(tensors: &BTreeMap<String, Vec<f32>>) -> io::Result<Vec<u8>> {
    if tensors.is_empty() {
        return Err(invalid_input("cannot serialize an empty checkpoint"));
    }
    let mut buf = Vec::new();
    for (id, values) in tensors {
        if id.is_empty() || id.len() > MAX_TENSOR_ID_BYTES || !id.is_ascii() || id.contains('\n') {
            return Err(invalid_input("checkpoint tensor id must be non-empty ASCII and at most 256 bytes"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("cannot serialize a non-finite weight"));
        }
        buf.extend_from_slice(id.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(&values.len().to_le_bytes());
        for value in values {
            buf.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(buf)
}

/// Elementwise mean of checkpoints with exactly the same tensor schema.
pub fn average(checkpoints: &[BTreeMap<String, Vec<f32>>]) -> io::Result<BTreeMap<String, Vec<f32>>> {
    let first = checkpoints.first().ok_or_else(|| invalid_input("average requires at least one checkpoint"))?;
    for (peer, other) in checkpoints.iter().enumerate().skip(1) {
        if other.len() != first.len() || other.keys().ne(first.keys()) {
            return Err(invalid_data(&format!("checkpoint {peer} has a different tensor schema")));
        }
    }

    let n = checkpoints.len() as f32;
    let mut out = BTreeMap::new();
    for (id, first_values) in first {
        let mut sum = first_values.clone();
        for (peer, other) in checkpoints.iter().enumerate().skip(1) {
            let other_values = &other[id];
            if other_values.len() != sum.len() {
                return Err(invalid_data(&format!(
                    "tensor `{id}` has mismatched length in checkpoint {peer} ({} vs {})",
                    other_values.len(), sum.len()
                )));
            }
            for (sum, value) in sum.iter_mut().zip(other_values) {
                *sum += value;
            }
        }
        for value in &mut sum {
            *value /= n;
            if !value.is_finite() {
                return Err(invalid_data(&format!("averaging tensor `{id}` produced a non-finite weight")));
            }
        }
        out.insert(id.clone(), sum);
    }
    Ok(out)
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(pairs: &[(&str, &[f32])]) -> Vec<u8> {
        let map: BTreeMap<String, Vec<f32>> =
            pairs.iter().map(|(id, values)| (id.to_string(), values.to_vec())).collect();
        serialize(&map).unwrap()
    }

    #[test]
    fn round_trips_through_parse_and_serialize() {
        let bytes = make(&[("l0w", &[1.0, 2.0, 3.0]), ("l0b", &[0.5])]);
        let parsed = parse(&bytes).unwrap();
        assert_eq!(parsed["l0w"], vec![1.0, 2.0, 3.0]);
        assert_eq!(parsed["l0b"], vec![0.5]);
        assert_eq!(serialize(&parsed).unwrap(), bytes);
    }

    #[test]
    fn averages_checkpoints_elementwise() {
        let a = parse(&make(&[("w", &[1.0, 0.0])])).unwrap();
        let b = parse(&make(&[("w", &[2.0, 3.0])])).unwrap();
        let c = parse(&make(&[("w", &[3.0, 9.0])])).unwrap();
        assert_eq!(average(&[a, b, c]).unwrap()["w"], vec![2.0, 4.0]);
    }

    #[test]
    fn rejects_duplicate_tensor_ids() {
        let mut bytes = make(&[("w", &[1.0])]);
        bytes.extend_from_slice(&make(&[("w", &[2.0])]));
        assert_eq!(parse(&bytes).unwrap_err().kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_extra_tensor_schema() {
        let a = parse(&make(&[("w", &[1.0])])).unwrap();
        let b = parse(&make(&[("w", &[1.0]), ("other", &[2.0])])).unwrap();
        assert_eq!(average(&[a, b]).unwrap_err().kind(), ErrorKind::InvalidData);
    }
}
