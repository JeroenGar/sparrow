use anyhow::Result;
use log::{Level, Log, Metadata, Record};
use sparrow::EPOCH;
use std::fs;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, SyncSender};

pub(super) struct LogEntry {
    pub(super) level: Level,
    pub(super) message: String,
}

struct TuiLogSink {
    logs: SyncSender<LogEntry>,
}

impl Log for TuiLogSink {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        let _ = self.logs.try_send(LogEntry {
            level: record.level(),
            message: record.args().to_string(),
        });
    }

    fn flush(&self) {}
}

pub(super) fn init(level: log::LevelFilter, log_file_path: &Path) -> Result<Receiver<LogEntry>> {
    let _ = fs::remove_file(log_file_path);
    let (logs, receiver) = mpsc::sync_channel(512);
    fern::Dispatch::new()
        .format(|out, message, record| {
            let elapsed = EPOCH.elapsed();
            let seconds = elapsed.as_secs() % 60;
            let minutes = (elapsed.as_secs() / 60) % 60;
            let hours = elapsed.as_secs() / 3600;
            out.finish(format_args!(
                "[{}] [{hours:02}:{minutes:02}:{seconds:02}] {}",
                record.level(),
                message,
            ));
        })
        .level(level)
        .chain(Box::new(TuiLogSink { logs }) as Box<dyn Log>)
        .chain(fern::log_file(log_file_path)?)
        .apply()?;
    Ok(receiver)
}
