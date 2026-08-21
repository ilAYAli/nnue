use std::{io, net::{TcpStream, ToSocketAddrs}, thread, time::{Duration, Instant}};

use super::protocol;

/// Connects to the coordinator, sends a round-tagged local checkpoint, and
/// waits for the matching averaged reply. Connection attempts are retried until
/// the shared round timeout, allowing workers to start before the coordinator.
pub fn sync(
    coordinator_addr: &str,
    run_id: &str,
    worker_id: &str,
    superbatch: usize,
    local_weights: &[u8],
    timeout: Duration,
) -> io::Result<Vec<u8>> {
    let deadline = Instant::now() + timeout;
    let mut last_error = None;
    while Instant::now() < deadline {
        match connect_before(coordinator_addr, deadline) {
            Ok(mut stream) => {
                stream.set_read_timeout(Some(remaining(deadline)?))?;
                stream.set_write_timeout(Some(remaining(deadline)?))?;
                protocol::write_request(&mut stream, &protocol::SyncRequest {
                    run_id: run_id.to_owned(), worker_id: worker_id.to_owned(), superbatch, checkpoint: local_weights.to_vec(),
                })?;
                let response = protocol::read_response(&mut stream)?;
                if response.superbatch != superbatch {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "coordinator replied for a different superbatch"));
                }
                return Ok(response.checkpoint);
            }
            Err(error) if retryable(&error) => {
                last_error = Some(error);
                thread::sleep(remaining(deadline)?.min(Duration::from_millis(200)));
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "distributed worker timed out connecting to coordinator")))
}

fn connect_before(address: &str, deadline: Instant) -> io::Result<TcpStream> {
    let addresses = address.to_socket_addrs()?;
    let mut last_error = None;
    for address in addresses {
        match TcpStream::connect_timeout(&address, remaining(deadline)?) {
            Ok(stream) => return Ok(stream),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "coordinator address resolved to no socket addresses")))
}

fn remaining(deadline: Instant) -> io::Result<Duration> {
    deadline.checked_duration_since(Instant::now()).ok_or_else(|| {
        io::Error::new(io::ErrorKind::TimedOut, "distributed worker timed out waiting for coordinator")
    })
}

fn retryable(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::ConnectionRefused | io::ErrorKind::ConnectionAborted | io::ErrorKind::ConnectionReset
            | io::ErrorKind::NotConnected | io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    )
}
