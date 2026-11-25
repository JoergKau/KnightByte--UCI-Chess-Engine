//! The diagnostic counters: `searchstats` and `nnueregions`.

use crate::helpers::{field, run_engine, run_in_two_phases};

#[test]
fn searchstats_is_silent_until_asked_and_reports_the_counted_search() {
    // Off by default: the report is all zeros and no `qshare` is invented.
    let idle = run_engine(b"searchstats\nquit\n");
    assert!(
        idle.contains("searchstats nodes 0 qnodes 0 qshare 0.0% tt 0 hits 0"),
        "unexpected idle report: {idle}"
    );

    let lines = run_in_two_phases(
        b"searchstats reset\nposition startpos\ngo depth 8\n",
        b"searchstats\n",
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("searchstats: counters cleared, counting STARTED"))
    );
    let last_info = lines
        .iter()
        .rfind(|l| l.starts_with("info depth 8 "))
        .expect("a depth-8 info line");
    let report = lines
        .iter()
        .rfind(|l| l.starts_with("searchstats nodes"))
        .expect("a searchstats report");
    // One thread: the report's node count is the search's own count.
    assert_eq!(field(report, "nodes"), field(last_info, "nodes"));
    let qnodes = field(report, "qnodes");
    assert!(qnodes > 0 && qnodes < field(report, "nodes"), "{report}");
    let probes = field(report, "tt");
    assert!(probes > 0 && field(report, "hits") <= probes, "{report}");
    assert!(field(report, "cutoffs") > 0, "{report}");

    // The second line describes how the main search prunes. Every main-search
    // node probes the table exactly once, so its main-node count equals the
    // first line's probe count; the depth histogram sums to the same number.
    let prune = lines
        .iter()
        .rfind(|l| l.starts_with("searchprune main"))
        .expect("a searchprune report");
    assert_eq!(field(prune, "main"), probes, "{prune}");
    assert!(field(prune, "moves") >= field(prune, "main"), "{prune}");
    assert!(
        field(prune, "rfp") > 0 && field(prune, "see") > 0,
        "{prune}"
    );
    let histogram: u64 = prune
        .rsplit(' ')
        .next()
        .expect("the histogram")
        .split('/')
        .map(|bucket| bucket.parse::<u64>().expect("a bucket count"))
        .sum();
    assert_eq!(histogram, probes, "{prune}");
}

#[test]
fn searchstats_off_freezes_the_snapshot() {
    let lines = run_in_two_phases(
        b"searchstats reset\nposition startpos\ngo depth 6\n",
        b"searchstats off\nposition startpos\ngo depth 4\nsearchstats\n",
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("searchstats: counting STOPPED"))
    );
    let reports: Vec<&String> = lines
        .iter()
        .filter(|l| l.starts_with("searchstats nodes"))
        .collect();
    // `reset` and `off` each print the current report too: reset (empty),
    // off (the counted search), and the final bare command.
    assert_eq!(reports.len(), 3, "three reports expected: {reports:?}");
    assert_ne!(reports[0], reports[1], "the counted search left no trace");
    assert_eq!(
        reports[1], reports[2],
        "a search with counting off changed the snapshot"
    );
}

#[test]
fn nnueregions_counts_refreshes_per_king_region_and_stops_on_off() {
    let idle = run_engine(b"nnueregions\nquit\n");
    assert!(
        idle.contains("nnueregions r0=0 r1=0 r2=0 r3=0 r4=0 r5=0 r6=0 r7=0 total 0\n")
            || idle.contains("nnueregions r0=0 r1=0 r2=0 r3=0 r4=0 r5=0 r6=0 r7=0 total 0\r\n"),
        "unexpected idle histogram: {idle}"
    );
    assert!(!idle.contains("shares"), "no shares line without counts");

    let lines = run_in_two_phases(
        b"nnueregions reset\nposition startpos\ngo depth 8\n",
        b"nnueregions\nnnueregions off\nposition startpos\ngo depth 4\nnnueregions\n",
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("nnueregions: counters cleared, counting STARTED"))
    );
    let histograms: Vec<&String> = lines
        .iter()
        .filter(|l| l.starts_with("nnueregions r0="))
        .collect();
    // reset (empty), the bare command, off, and the final bare command.
    assert_eq!(histograms.len(), 4, "{histograms:?}");
    assert_eq!(field(histograms[0], "total"), 0);
    let total = field(histograms[1], "total");
    assert!(total > 0, "no refresh was counted: {}", histograms[1]);
    assert!(histograms[1].ends_with("(counting)"));
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("info string nnueregions shares: r0 "))
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("nnueregions: counting STOPPED"))
    );
    // After `off` the later search adds nothing and the marker is gone.
    assert_eq!(field(histograms[2], "total"), total);
    assert_eq!(field(histograms[3], "total"), total);
    assert!(!histograms[2].ends_with("(counting)"));
    assert!(!histograms[3].ends_with("(counting)"));
}
// The old "an unsupported go limiter still gets a bestmove" guarantee is
// fully covered since step 6: every `go` limiter is implemented (ponder,
// mate, searchmoves included), and every `go` is answered by its worker —
// see the step 4/5/6 tests below.
