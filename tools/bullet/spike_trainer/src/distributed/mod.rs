//! Periodic weight-averaging (local SGD) for a small trusted GPU fleet.
//! Each machine trains locally and exchanges model weights only at configured
//! superbatch boundaries, avoiding per-batch network synchronization.

mod checkpoint;
mod coordinator;
mod protocol;
mod worker;

use std::{io, net::TcpListener, time::Duration};

use bullet_lib::nn::ExecutionContext;
use bullet_trainer::model::Model;

pub enum Role {
    /// Accepts `num_peers` workers, averages their weights with its own, and
    /// broadcasts the average to every worker.
    Coordinator { listen_addr: String, num_peers: usize },
    /// Connects to the coordinator for every synchronization round.
    Worker { coordinator_addr: String },
}

pub struct DistributedConfig {
    pub role: Role,
    /// A stable identity shared by every member of one distributed run.
    pub run_id: String,
    /// The unique hostname/identity of this process. Only worker identities
    /// are sent on the wire, but validating it for all members avoids silent
    /// misconfiguration.
    pub node_id: String,
    /// Average weights every this many superbatches. Must be at least one.
    pub sync_every_superbatches: usize,
    /// Maximum time a synchronization round may wait for peers or I/O.
    pub round_timeout: Duration,
}

/// Builds the per-superbatch hook consumed by `ValueTrainer::run_distributed`.
/// The coordinator listener is bound before training begins so workers can
/// retry safely while their coordinator finishes initialization.
pub fn make_sync_hook(
    config: DistributedConfig,
    final_superbatch: usize,
) -> io::Result<impl FnMut(&mut Model<ExecutionContext>, usize) -> bool> {
    validate_config(&config)?;
    let coordinator_peers = match &config.role {
        Role::Coordinator { num_peers, .. } => Some(*num_peers),
        Role::Worker { .. } => None,
    };
    let listener = match &config.role {
        Role::Coordinator { listen_addr, .. } => {
            let listener = TcpListener::bind(listen_addr)?;
            listener.set_nonblocking(true)?;
            eprintln!(
                "distributed coordinator ready: run={} listen={} workers={} sync_every={} timeout={}s",
                config.run_id,
                listener.local_addr()?,
                coordinator_peers.expect("coordinator has a peer count"),
                config.sync_every_superbatches,
                config.round_timeout.as_secs(),
            );
            Some(listener)
        }
        Role::Worker { coordinator_addr } => {
            eprintln!(
                "distributed worker ready: run={} node={} coordinator={} sync_every={} timeout={}s",
                config.run_id,
                config.node_id,
                coordinator_addr,
                config.sync_every_superbatches,
                config.round_timeout.as_secs(),
            );
            None
        }
    };

    Ok(move |model: &mut Model<ExecutionContext>, superbatch: usize| {
        // The final model must be a fleet-wide average even if the requested
        // cadence does not divide the training length. Otherwise exporting on
        // the coordinator would silently discard the workers' final updates.
        if superbatch % config.sync_every_superbatches != 0 && superbatch != final_superbatch {
            return false;
        }

        let mut local = Vec::new();
        model.write_to(&mut local).expect("serialize local weights for distributed sync");
        eprintln!("distributed sync: run={} superbatch={superbatch}", config.run_id);

        let averaged = match &config.role {
            Role::Worker { coordinator_addr } => worker::sync(
                coordinator_addr,
                &config.run_id,
                &config.node_id,
                superbatch,
                &local,
                config.round_timeout,
            )
            .unwrap_or_else(|err| panic!("distributed sync with coordinator {coordinator_addr} failed: {err}")),
            Role::Coordinator { num_peers, .. } => {
                let listener = listener.as_ref().expect("coordinator listener bound at hook creation");
                coordinator::run_round(
                    listener,
                    *num_peers,
                    &config.run_id,
                    superbatch,
                    &local,
                    config.round_timeout,
                )
                .unwrap_or_else(|err| panic!("distributed sync round failed: {err}"))
            }
        };

        model.load_from(&averaged[..]).expect("load averaged weights back into model");
        true
    })
}

fn validate_config(config: &DistributedConfig) -> io::Result<()> {
    if config.run_id.is_empty() || !config.run_id.is_ascii() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed run id must be non-empty ASCII"));
    }
    if config.node_id.is_empty() || !config.node_id.is_ascii() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed node id must be non-empty ASCII"));
    }
    if config.sync_every_superbatches == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed sync interval must be at least one superbatch"));
    }
    if config.round_timeout.is_zero() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed round timeout must be positive"));
    }
    match &config.role {
        Role::Coordinator { listen_addr, num_peers } => {
            if listen_addr.is_empty() || *num_peers == 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed coordinator requires a listen address and at least one worker"));
            }
        }
        Role::Worker { coordinator_addr } if coordinator_addr.is_empty() => {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "distributed worker requires a coordinator address"));
        }
        Role::Worker { .. } => {}
    }
    Ok(())
}
