use std::io::{self, ErrorKind, Read, Write};

pub const VERSION: u16 = 1;
pub const MAX_FRAME_BYTES: usize = 512 * 1024 * 1024;
const MAX_ID_BYTES: usize = 256;

#[derive(Debug, PartialEq, Eq)]
pub struct SyncRequest {
    pub run_id: String,
    pub worker_id: String,
    pub superbatch: usize,
    pub checkpoint: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SyncResponse {
    pub superbatch: usize,
    pub checkpoint: Vec<u8>,
}

pub fn write_request(stream: &mut impl Write, request: &SyncRequest) -> io::Result<()> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&VERSION.to_le_bytes());
    write_string(&mut payload, &request.run_id)?;
    write_string(&mut payload, &request.worker_id)?;
    payload.extend_from_slice(&request.superbatch.to_le_bytes());
    payload.extend_from_slice(&request.checkpoint);
    write_frame(stream, &payload)
}

pub fn read_request(stream: &mut impl Read) -> io::Result<SyncRequest> {
    let payload = read_frame(stream)?;
    let mut cursor = payload.as_slice();
    require_version(&mut cursor)?;
    let run_id = read_string(&mut cursor)?;
    let worker_id = read_string(&mut cursor)?;
    let superbatch = read_usize(&mut cursor)?;
    if cursor.is_empty() {
        return Err(invalid_data("sync request has an empty checkpoint"));
    }
    Ok(SyncRequest { run_id, worker_id, superbatch, checkpoint: cursor.to_vec() })
}

pub fn write_response(stream: &mut impl Write, response: &SyncResponse) -> io::Result<()> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&VERSION.to_le_bytes());
    payload.extend_from_slice(&response.superbatch.to_le_bytes());
    payload.extend_from_slice(&response.checkpoint);
    write_frame(stream, &payload)
}

pub fn read_response(stream: &mut impl Read) -> io::Result<SyncResponse> {
    let payload = read_frame(stream)?;
    let mut cursor = payload.as_slice();
    require_version(&mut cursor)?;
    let superbatch = read_usize(&mut cursor)?;
    if cursor.is_empty() {
        return Err(invalid_data("sync response has an empty checkpoint"));
    }
    Ok(SyncResponse { superbatch, checkpoint: cursor.to_vec() })
}

fn write_string(payload: &mut Vec<u8>, value: &str) -> io::Result<()> {
    if value.is_empty() || value.len() > MAX_ID_BYTES || !value.is_ascii() {
        return Err(invalid_input("distributed identifier must be non-empty ASCII and at most 256 bytes"));
    }
    payload.extend_from_slice(&(value.len() as u16).to_le_bytes());
    payload.extend_from_slice(value.as_bytes());
    Ok(())
}

fn read_string(cursor: &mut &[u8]) -> io::Result<String> {
    let len = read_u16(cursor)? as usize;
    if len == 0 || len > MAX_ID_BYTES || cursor.len() < len {
        return Err(invalid_data("invalid distributed identifier"));
    }
    let (bytes, rest) = cursor.split_at(len);
    *cursor = rest;
    if !bytes.is_ascii() {
        return Err(invalid_data("distributed identifier is not ASCII"));
    }
    String::from_utf8(bytes.to_vec()).map_err(|_| invalid_data("distributed identifier is not UTF-8"))
}

fn require_version(cursor: &mut &[u8]) -> io::Result<()> {
    let version = read_u16(cursor)?;
    if version != VERSION {
        return Err(invalid_data(&format!("unsupported distributed protocol version {version}")));
    }
    Ok(())
}

fn read_u16(cursor: &mut &[u8]) -> io::Result<u16> {
    let bytes = take(cursor, 2)?;
    Ok(u16::from_le_bytes(bytes.try_into().expect("fixed-size slice")))
}

fn read_usize(cursor: &mut &[u8]) -> io::Result<usize> {
    let bytes = take(cursor, size_of::<usize>())?;
    Ok(usize::from_le_bytes(bytes.try_into().expect("fixed-size slice")))
}

fn take<'a>(cursor: &mut &'a [u8], len: usize) -> io::Result<&'a [u8]> {
    if cursor.len() < len {
        return Err(invalid_data("truncated distributed protocol message"));
    }
    let (head, rest) = cursor.split_at(len);
    *cursor = rest;
    Ok(head)
}

fn write_frame(stream: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    if payload.len() > MAX_FRAME_BYTES {
        return Err(invalid_input("distributed frame exceeds maximum size"));
    }
    stream.write_all(&(payload.len() as u32).to_le_bytes())?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_frame(stream: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(invalid_data("distributed frame exceeds maximum size"));
    }

    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok(payload)
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
    use std::io::Cursor;

    #[test]
    fn round_trips_request_and_response() {
        let request = SyncRequest {
            run_id: "enyo-16.0.0-rc2".to_owned(),
            worker_id: "pwa-5090".to_owned(),
            superbatch: 64,
            checkpoint: b"local weights".to_vec(),
        };
        let mut request_buf = Cursor::new(Vec::new());
        write_request(&mut request_buf, &request).unwrap();
        request_buf.set_position(0);
        assert_eq!(read_request(&mut request_buf).unwrap(), request);

        let response = SyncResponse { superbatch: 64, checkpoint: b"averaged weights".to_vec() };
        let mut response_buf = Cursor::new(Vec::new());
        write_response(&mut response_buf, &response).unwrap();
        response_buf.set_position(0);
        assert_eq!(read_response(&mut response_buf).unwrap(), response);
    }

    #[test]
    fn rejects_oversized_frame_before_allocating() {
        let mut buf = Cursor::new((MAX_FRAME_BYTES as u32 + 1).to_le_bytes().to_vec());
        let error = read_request(&mut buf).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_empty_identifiers() {
        let mut buf = Cursor::new(Vec::new());
        let error = write_request(
            &mut buf,
            &SyncRequest {
                run_id: String::new(),
                worker_id: "pwa-hak".to_owned(),
                superbatch: 1,
                checkpoint: vec![1],
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
