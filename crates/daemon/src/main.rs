//! Console entry point for the dictation engine. See `daemon` (this
//! crate's `lib.rs`) for the actual pipeline -- this binary just loads it,
//! prints progress to stdout, and lets it run until Ctrl+C.
//!
//! For a GUI front end with a system tray icon and a floating recording
//! indicator, see `crates/tray-app`, which drives the same `daemon::Engine`.

use std::sync::mpsc;

fn main() {
    println!("Local Dictation Engine -- console mode");

    let engine = match daemon::Engine::load(|line| println!("{line}")) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("See the relevant crate's README.md for how to fetch a model file.");
            std::process::exit(1);
        }
    };

    println!("Ctrl+C to quit.");

    // The console binary doesn't consume structured status updates -- all
    // of engine.run()'s progress already goes to stdout via println!.
    // Sends on a receiver-less channel are harmless no-ops (checked with
    // `let _ =` throughout daemon::Engine::run), so dropping the receiver
    // immediately is fine; no drain thread needed.
    let (status_tx, _status_rx) = mpsc::channel();
    engine.run(status_tx);
}
