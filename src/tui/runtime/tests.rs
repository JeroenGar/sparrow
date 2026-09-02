use super::*;

#[test]
fn phase_interrupt_resets_but_quit_does_not() {
    let signals = TuiSignals::new();
    signals.interrupt_phase();
    let mut terminator = TuiTerminator::new(signals.clone());

    assert!(terminator.kill());
    terminator.new_timeout(Duration::from_secs(1));
    assert!(!terminator.kill());

    signals.request_quit();
    terminator.new_timeout(Duration::from_secs(1));
    assert!(terminator.kill());
}
