//! Pairwise ranking and the train/validation split.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{EvolveError, Result};
use crate::judge::{build_bundle, PairwiseJudge};
use crate::ranking::{bradley_terry, tournament_pairs, Match};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub name: String,
    pub task: u32,
    pub condition: String,
    pub seed: u64,
    pub workspace: PathBuf,
    pub logs_dir: PathBuf,
    #[serde(default)]
    pub teacher_prompt: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedEpisode {
    pub name: String,
    pub task: u32,
    pub condition: String,
    pub seed: u64,
    pub workspace: PathBuf,
    pub logs_dir: PathBuf,
    #[serde(default)]
    pub teacher_prompt: Option<PathBuf>,
    pub score: f64,
    pub rank: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    pub train: Vec<RankedEpisode>,
    pub validation: RankedEpisode,
}

impl Candidate {
    pub fn ranked(self, score: f64, rank: u32) -> RankedEpisode {
        RankedEpisode {
            name: self.name,
            task: self.task,
            condition: self.condition,
            seed: self.seed,
            workspace: self.workspace,
            logs_dir: self.logs_dir,
            teacher_prompt: self.teacher_prompt,
            score,
            rank,
        }
    }
}

pub fn rank_candidates(
    judge: &dyn PairwiseJudge,
    task_text: &str,
    candidates: &[Candidate],
    opponents: usize,
    seed: u64,
    bt_iterations: usize,
) -> Result<Vec<RankedEpisode>> {
    if candidates.is_empty() {
        return Err(EvolveError::Ranking("no candidates to rank".into()));
    }
    let names: Vec<String> = candidates.iter().map(|value| value.name.clone()).collect();
    let pairs = tournament_pairs(&names, opponents, seed)?;

    let mut by_name = std::collections::BTreeMap::new();
    for candidate in candidates {
        if by_name.insert(candidate.name.as_str(), candidate).is_some() {
            return Err(EvolveError::Ranking(format!(
                "candidate {:?} appears twice",
                candidate.name
            )));
        }
    }

    let mut matches = Vec::with_capacity(pairs.len());
    for (index, (a, b)) in pairs.iter().enumerate() {
        let left = by_name
            .get(a.as_str())
            .ok_or_else(|| EvolveError::Ranking(format!("unknown candidate {a:?}")))?;
        let right = by_name
            .get(b.as_str())
            .ok_or_else(|| EvolveError::Ranking(format!("unknown candidate {b:?}")))?;
        let bundle_a = build_bundle(&left.workspace, task_text, Some(&left.logs_dir))?;
        let bundle_b = build_bundle(&right.workspace, task_text, Some(&right.logs_dir))?;
        let verdict = judge.compare(
            task_text,
            &bundle_a,
            &bundle_b,
            seed.wrapping_add(index as u64),
        )?;
        matches.push(Match {
            a: a.clone(),
            b: b.clone(),
            winner: verdict.winner,
        });
    }

    let scores = bradley_terry(&names, &matches, bt_iterations, 0.5)?;
    let mut ranked: Vec<RankedEpisode> = candidates
        .iter()
        .cloned()
        .map(|candidate| {
            let score = scores.get(&candidate.name).copied().unwrap_or(0.0);
            candidate.ranked(score, 0)
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.name.cmp(&right.name))
    });
    for (index, row) in ranked.iter_mut().enumerate() {
        row.rank = index as u32 + 1;
    }
    Ok(ranked)
}

pub fn select_rows(
    ranked: &[RankedEpisode],
    train_ranks: u32,
    validation_rank: u32,
) -> Result<Selection> {
    if validation_rank != train_ranks + 1 {
        return Err(EvolveError::Ranking(
            "validation_rank must immediately follow train_ranks".into(),
        ));
    }
    let mut train = Vec::new();
    let mut validation = None;
    for row in ranked {
        if row.rank >= 1 && row.rank <= train_ranks {
            train.push(row.clone());
        }
        if row.rank == validation_rank {
            validation = Some(row.clone());
        }
    }
    let validation = validation.ok_or_else(|| {
        EvolveError::Ranking(format!("missing validation rank {validation_rank}"))
    })?;
    if train.is_empty() {
        return Err(EvolveError::Ranking("training split is empty".into()));
    }
    if train.iter().any(|row| row.name == validation.name) {
        return Err(EvolveError::Ranking(
            "training and validation overlap".into(),
        ));
    }
    Ok(Selection { train, validation })
}

/// Build a candidate from a pool row.
pub fn candidate_from_pool(
    name: impl Into<String>,
    task: u32,
    condition: impl Into<String>,
    seed: u64,
    workspace: impl Into<PathBuf>,
    logs_dir: impl Into<PathBuf>,
    teacher_prompt: Option<PathBuf>,
) -> Candidate {
    Candidate {
        name: name.into(),
        task,
        condition: condition.into(),
        seed,
        workspace: workspace.into(),
        logs_dir: logs_dir.into(),
        teacher_prompt,
    }
}

pub fn write_selection(directory: &Path, selection: &Selection) -> Result<()> {
    crate::io::write_json(&directory.join("train.json"), &selection.train)?;
    crate::io::write_json(&directory.join("val.json"), &selection.validation)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{Verdict, Winner, WorkspaceBundle};

    struct AlwaysA;

    impl PairwiseJudge for AlwaysA {
        fn compare(
            &self,
            _task: &str,
            _a: &WorkspaceBundle,
            _b: &WorkspaceBundle,
            _seed: u64,
        ) -> Result<Verdict> {
            Ok(Verdict {
                winner: Winner::A,
                rationale: "a".into(),
            })
        }
    }

    fn candidate(name: &str, workspace: &Path) -> Candidate {
        Candidate {
            name: name.into(),
            task: 1,
            condition: "t1_L0".into(),
            seed: 1,
            workspace: workspace.to_path_buf(),
            logs_dir: workspace.join("logs"),
            teacher_prompt: None,
        }
    }

    #[test]
    fn ranks_a_four_candidate_pool() {
        let root = std::env::temp_dir().join(format!(
            "evolve-rank-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut candidates = Vec::new();
        for index in 0..4 {
            let workspace = root.join(format!("w{index}"));
            std::fs::create_dir_all(&workspace).unwrap();
            std::fs::write(workspace.join("result.md"), format!("# {index}\n")).unwrap();
            candidates.push(candidate(&format!("c{index}"), &workspace));
        }
        let ranked = rank_candidates(&AlwaysA, "do it", &candidates, 2, 7, 500).expect("rank");
        assert_eq!(ranked.len(), 4);
        assert_eq!(ranked[0].rank, 1);
        assert!(ranked.windows(2).all(|pair| pair[0].score >= pair[1].score));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selects_three_train_and_one_validation() {
        let root = std::env::temp_dir().join(format!(
            "evolve-select-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut ranked = Vec::new();
        for index in 1..=4 {
            let mut row = candidate(&format!("c{index}"), &root).ranked(0.0, index);
            row.score = -(index as f64);
            ranked.push(row);
        }
        let selection = select_rows(&ranked, 3, 4).expect("select");
        assert_eq!(selection.train.len(), 3);
        assert_eq!(selection.validation.rank, 4);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_a_missing_validation_rank() {
        let mut row = candidate("c1", Path::new("/tmp")).ranked(0.0, 1);
        row.rank = 1;
        assert!(select_rows(&[row], 3, 4).is_err());
    }
}
