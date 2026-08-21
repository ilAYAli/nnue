use std::{collections::BTreeSet, io, net::{TcpListener, TcpStream}, thread, time::{Duration, Instant}};

use super::{checkpoint, protocol};

/// Waits for exactly `num_peers` distinct workers for one round, averages them
/// with `local_weights`, and sends the same result to every worker.
pub fn run_round(
    listener: &TcpListener,
    num_peers: usize,
    run_id: &str,
    superbatch: usize,
    local_weights: &[u8],
    timeout: Duration,
) -> io::Result<Vec<u8>> {
    let deadline = Instant::now() + timeout;
    let mut peer_streams = Vec::with_capacity(num_peers);
    let mut peer_ids = BTreeSet::new();
    let mut parsed = vec![checkpoint::parse(local_weights)?];

    while peer_streams.len() < num_peers {
        let mut stream = accept_before(listener, deadline)?;
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        let request = protocol::read_request(&mut stream)?;
        if request.run_id != run_id {
            return Err(invalid_data("worker belongs to a different distributed run"));
        }
        if request.superbatch != superbatch {
            return Err(invalid_data("worker is at a different superbatch"));
        }
        if !peer_ids.insert(request.worker_id.clone()) {
            return Err(invalid_data("duplicate distributed worker identity"));
        }
        parsed.push(checkpoint::parse(&request.checkpoint)?);
        peer_streams.push(stream);
    }

    let averaged = checkpoint::serialize(&checkpoint::average(&parsed)?)?;
    let response = protocol::SyncResponse { superbatch, checkpoint: averaged.clone() };
    for mut stream in peer_streams {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        protocol::write_response(&mut stream, &response)?;
    }
    Ok(averaged)
}

fn accept_before(listener: &TcpListener, deadline: Instant) -> io::Result<TcpStream> {
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                let wait = remaining(deadline)?.min(Duration::from_millis(20));
                thread::sleep(wait);
            }
            Err(error) => return Err(error),
        }
    }
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline.checked_duration_since(Instant::now()).ok_or_else(|| {
        io::Error::new(io::ErrorKind::TimedOut, "distributed sync round timed out waiting for peers")
    })
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpStream, thread};

    fn checkpoint(values: &[f32]) -> Vec<u8> {
        checkpoint::serialize(&[("w".to_string(), values.to_vec())].into_iter().collect()).unwrap()
    }

    #[test]
    fn averages_local_plus_two_distinct_peers() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let peer = |worker_id: &'static str, values: &'static [f32]| thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            protocol::write_request(&mut stream, &protocol::SyncRequest {
                run_id: "test-run".to_owned(), worker_id: worker_id.to_owned(), superbatch: 8, checkpoint: checkpoint(values),
            }).unwrap();
            protocol::read_response(&mut stream).unwrap().checkpoint
        });
        let first = peer("pwa-5090", &[3.0, 0.0]);
        let second = peer("pwa-hak", &[9.0, 9.0]);
        let expected = checkpoint(&[4.0, 4.0]);
        assert_eq!(run_round(&listener, 2, "test-run", 8, &checkpoint(&[0.0, 3.0]), Duration::from_secs(1)).unwrap(), expected);
        assert_eq!(first.join().unwrap(), expected);
        assert_eq!(second.join().unwrap(), expected);
    }
}
