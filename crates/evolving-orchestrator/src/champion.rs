//! Champion derivation.
//!
//! The champion advances only on a direct win over the reigning champion.
//! Comparison history is the only input; no workspace or model is touched.

use crate::history::HistoryRow;
use crate::judge::Winner;

/// Adjacent comparison mode: `v{k-1}_vs_v{k}`.
///
/// Returns `(champion, arbitration_needed_for_current)`.
pub fn derive_champion(
    adjacent_rows: &[HistoryRow],
    champion_rows: &[HistoryRow],
    current_version: u32,
) -> (u32, bool) {
    let adjacent = adjacent_winners(adjacent_rows);
    let arbitration = champion_winners(champion_rows);

    let mut champion = 0;
    let mut needs_arbitration = false;
    for version in 1..=current_version {
        if adjacent.get(&version) != Some(&Winner::B) {
            continue;
        }
        if champion == version - 1 {
            champion = version;
        } else if let Some(winner) = arbitration.get(&(champion, version)) {
            if *winner == Winner::B {
                champion = version;
            }
        } else if version == current_version {
            needs_arbitration = true;
        }
    }
    (champion, needs_arbitration)
}

/// Fixed-reference mode: `v{k}_vs_no_harness`.
///
/// Returns `(champion, champion_has_baseline_win, arbitration_needed)`.
pub fn derive_champion_baseline(
    history_rows: &[HistoryRow],
    champion_rows: &[HistoryRow],
    current_version: u32,
) -> (u32, bool, bool) {
    let baseline = baseline_winners(history_rows);
    let arbitration = champion_winners(champion_rows);

    let mut champion = 0;
    let mut champion_has_win = baseline.get(&0) == Some(&Winner::B);
    let mut needs_arbitration = false;

    for version in 1..=current_version {
        if baseline.get(&version) != Some(&Winner::B) {
            continue;
        }
        if !champion_has_win {
            champion = version;
            champion_has_win = true;
        } else if let Some(winner) = arbitration.get(&(champion, version)) {
            if *winner == Winner::B {
                champion = version;
            }
        } else if version == current_version {
            needs_arbitration = true;
        }
    }
    (champion, champion_has_win, needs_arbitration)
}

fn adjacent_winners(rows: &[HistoryRow]) -> std::collections::BTreeMap<u32, Winner> {
    let mut out = std::collections::BTreeMap::new();
    for row in rows {
        if let Some(current) = parse_adjacent_key(&row.key) {
            out.insert(current, row.winner);
        }
    }
    out
}

fn baseline_winners(rows: &[HistoryRow]) -> std::collections::BTreeMap<u32, Winner> {
    let mut out = std::collections::BTreeMap::new();
    for row in rows {
        if let Some(version) = parse_baseline_key(&row.key) {
            out.insert(version, row.winner);
        }
    }
    out
}

fn champion_winners(rows: &[HistoryRow]) -> std::collections::BTreeMap<(u32, u32), Winner> {
    let mut out = std::collections::BTreeMap::new();
    for row in rows {
        if let Some((champion, current)) = parse_champion_key(&row.key) {
            out.insert((champion, current), row.winner);
        }
    }
    out
}

fn parse_adjacent_key(key: &str) -> Option<u32> {
    let rest = key.strip_prefix('v')?;
    let (previous, current) = rest.split_once("_vs_v")?;
    let previous: u32 = previous.parse().ok()?;
    let current: u32 = current.parse().ok()?;
    (current == previous + 1).then_some(current)
}

fn parse_baseline_key(key: &str) -> Option<u32> {
    let rest = key.strip_prefix('v')?;
    let version = rest.strip_suffix("_vs_no_harness")?;
    version.parse().ok()
}

fn parse_champion_key(key: &str) -> Option<(u32, u32)> {
    let rest = key.strip_prefix("champion_v")?;
    let (champion, current) = rest.split_once("_vs_v")?;
    Some((champion.parse().ok()?, current.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adj(previous: u32, current: u32, winner: Winner) -> HistoryRow {
        HistoryRow::adjacent(previous, current, winner, "", "", "", "")
    }

    fn champ(champion: u32, current: u32, winner: Winner) -> HistoryRow {
        HistoryRow::champion(champion, current, winner, "", "", "", "")
    }

    fn base(version: u32, winner: Winner) -> HistoryRow {
        HistoryRow::baseline(version, winner, "", "", "", "")
    }

    #[test]
    fn adjacent_win_advances_when_champion_is_predecessor() {
        let rows = vec![adj(0, 1, Winner::B), adj(1, 2, Winner::B)];
        assert_eq!(derive_champion(&rows, &[], 2), (2, false));
    }

    #[test]
    fn older_champion_requires_arbitration() {
        let rows = vec![
            adj(0, 1, Winner::B),
            adj(1, 2, Winner::A),
            adj(2, 3, Winner::B),
        ];
        let (champion, needs) = derive_champion(&rows, &[], 3);
        assert_eq!(champion, 1);
        assert!(needs);
    }

    #[test]
    fn arbitration_can_advance() {
        let rows = vec![
            adj(0, 1, Winner::B),
            adj(1, 2, Winner::A),
            adj(2, 3, Winner::B),
        ];
        let champion_rows = vec![champ(1, 3, Winner::B)];
        assert_eq!(derive_champion(&rows, &champion_rows, 3), (3, false));
    }

    #[test]
    fn baseline_mode_defaults_to_v0_without_a_win() {
        let (champion, has_win, needs) = derive_champion_baseline(&[], &[], 0);
        assert_eq!((champion, has_win, needs), (0, false, false));
    }

    #[test]
    fn baseline_win_takes_the_crown() {
        let rows = vec![base(0, Winner::A), base(1, Winner::B)];
        assert_eq!(derive_champion_baseline(&rows, &[], 1), (1, true, false));
    }

    #[test]
    fn second_baseline_winner_needs_arbitration() {
        let rows = vec![base(1, Winner::B), base(2, Winner::B)];
        assert_eq!(derive_champion_baseline(&rows, &[], 2), (1, true, true));
    }
}
