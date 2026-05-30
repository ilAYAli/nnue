use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    mem,
    process,
    time::Instant,
};

use bullet_lib::{
    game::formats::bulletformat::ChessBoard,
    value::loader::{
        DataLoader, SfBinpackLoader,
        sfbinpack::{MoveType, PieceType, TrainingDataEntry},
    },
};

#[derive(Clone, Copy)]
struct Filter {
    min_ply: u16,
    max_abs_cp: u32,
    quiet_only: bool,
}

impl Filter {
    fn keep(self, entry: &TrainingDataEntry) -> bool {
        if entry.ply < self.min_ply {
            return false;
        }
        if i32::from(entry.score).unsigned_abs() > self.max_abs_cp {
            return false;
        }
        if self.quiet_only {
            if entry.mv.mtype() != MoveType::Normal
                || entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None
            {
                return false;
            }
        }
        if entry.pos.is_checked(entry.pos.side_to_move()) {
            return false;
        }
        true
    }
}

fn write_chunk(writer: &mut BufWriter<File>, chunk: &[ChessBoard]) -> std::io::Result<()> {
    // ChessBoard is repr(C); this matches BulletFormat::write_to_bin exactly.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            chunk.as_ptr() as *const u8,
            chunk.len() * mem::size_of::<ChessBoard>(),
        )
    };
    writer.write_all(bytes)
}

fn usage() -> ! {
    eprintln!(
        "usage: convert_sfbinpack --data PATH[;PATH...] --output PATH \
         [--buffer-mb N] [--threads N] [--min-ply N] [--max-abs-cp N] [--quiet-only 0|1]"
    );
    process::exit(2);
}

fn main() {
    let mut data = String::new();
    let mut output = String::new();
    let mut buffer_mb = 1024_usize;
    let mut threads = 4_usize;
    let mut filter = Filter {
        min_ply: 16,
        max_abs_cp: 10000,
        quiet_only: true,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--data"       => data = value(),
            "--output"     => output = value(),
            "--buffer-mb"  => buffer_mb  = value().parse().unwrap_or_else(|_| usage()),
            "--threads"    => threads    = value().parse().unwrap_or_else(|_| usage()),
            "--min-ply"    => filter.min_ply    = value().parse().unwrap_or_else(|_| usage()),
            "--max-abs-cp" => filter.max_abs_cp = value().parse().unwrap_or_else(|_| usage()),
            "--quiet-only" => filter.quiet_only = value() != "0",
            _ => usage(),
        }
    }

    if data.is_empty() || output.is_empty() {
        usage();
    }

    let paths: Vec<&str> = data
        .split(';')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if paths.is_empty() {
        usage();
    }

    let out_file = File::create(&output).unwrap_or_else(|e| {
        eprintln!("error: cannot create {output}: {e}");
        process::exit(1);
    });
    let mut writer = BufWriter::new(out_file);

    let loader = SfBinpackLoader::new_concat_multiple(&paths, buffer_mb, threads, move |entry| {
        filter.keep(entry)
    });

    let start = Instant::now();
    let mut written: u64 = 0;

    loader.map_chunks(0, |chunk: &[ChessBoard]| {
        write_chunk(&mut writer, chunk).unwrap_or_else(|e| {
            eprintln!("error: write failed: {e}");
            process::exit(1);
        });
        written += chunk.len() as u64;
        if written % 5_000_000 < chunk.len() as u64 {
            let secs = start.elapsed().as_secs_f32();
            eprintln!(
                "converted {} positions ({:.1}M/s)",
                written,
                written as f32 / secs.max(0.001) / 1e6
            );
        }
        false
    });

    let secs = start.elapsed().as_secs_f32();
    eprintln!(
        "done: converted {} positions to {} in {:.1}s ({:.1}M/s)",
        written,
        output,
        secs,
        written as f32 / secs.max(0.001) / 1e6,
    );
}
