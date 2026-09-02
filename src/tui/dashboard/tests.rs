use super::*;

fn dashboard() -> Dashboard {
    Dashboard::new(SearchBudget {
        total_duration: Duration::ZERO,
        max_attempts: None,
        shrink_range: None,
        exploration_boundary: None,
    })
}

#[test]
fn groups_separation_attempts_by_width() {
    let mut dashboard = dashboard();
    let progress = |strip_width, iteration, min_loss| {
        DashboardUpdate::Separation(SeparationProgress {
            strip_width,
            density: 80.0,
            iteration,
            min_loss,
        })
    };

    dashboard.apply(progress(100.0, 0, 20.0));
    dashboard.apply(progress(100.0, 1, 10.0));
    assert_eq!(dashboard.attempt, 1);
    assert_eq!(dashboard.loss_remaining, Some(50.0));

    dashboard.apply(progress(100.0, 0, 15.0));
    assert_eq!(dashboard.attempt, 2);
    assert_eq!(dashboard.loss_remaining, Some(100.0));

    dashboard.apply(progress(99.0, 0, 12.0));
    assert_eq!(dashboard.attempt, 1);
}

#[test]
fn flashes_separation_results() {
    let mut dashboard = dashboard();
    let result = |success| SeparationResult {
        success,
        elapsed_seconds: 1.0,
        total_evals: 0,
        total_moves: 0,
        iterations: 0,
    };

    dashboard.apply(DashboardUpdate::SeparationResult(result(true)));
    assert_eq!(dashboard.separation_result_color(), Some(COLOR_ACCENT));

    dashboard.apply(DashboardUpdate::SeparationResult(result(false)));
    assert_eq!(dashboard.separation_result_color(), Some(COLOR_FAILURE));

    dashboard.last_separation_result = Some((result(true), Instant::now() - RESULT_FLASH_DURATION));
    assert_eq!(dashboard.separation_result_color(), None);
}

#[test]
fn keeps_scrolled_logs_in_place_as_new_lines_arrive() {
    let mut dashboard = dashboard();
    dashboard.log_view_height = 2;
    for line in 0..4 {
        dashboard.push_log(LogEntry {
            level: Level::Info,
            message: line.to_string(),
        });
    }

    dashboard.scroll_logs_up(usize::MAX);
    assert_eq!(dashboard.log_scroll, 2);
    dashboard.push_log(LogEntry {
        level: Level::Info,
        message: "new".to_owned(),
    });
    assert_eq!(dashboard.log_scroll, 3);
    dashboard.scroll_logs_down(usize::MAX);
    assert_eq!(dashboard.log_scroll, 0);
}

#[test]
fn wraps_log_lines_to_the_view_width() {
    assert_eq!(
        wrapped_log_lines("abcdefgh", 3).collect::<Vec<_>>(),
        vec!["abc", "def", "gh"]
    );
    assert_eq!(wrapped_log_lines("", 3).collect::<Vec<_>>(), vec![""]);
}

#[test]
fn shrink_progress_runs_from_initial_to_final_step() {
    let range = (0.05, 0.01);

    assert_eq!(shrink_progress(range.0, range), 0.0);
    assert_eq!(shrink_progress(range.1, range), 1.0);
    assert!((shrink_progress(0.03, range) - 0.5).abs() < 1e-6);
}
