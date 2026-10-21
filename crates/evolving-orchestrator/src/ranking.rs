//! Tournament scheduling and Bradley–Terry ranking.
//!
//! The schedule and the random draws must match CPython's `random.Random`
//! for a fixed seed, otherwise pair sets and scores differ between runs. The
//! generator below is a faithful MT19937 port with CPython's `_randbelow`
//! and `shuffle` behavior.

use std::collections::BTreeMap;

use crate::error::{EvolveError, Result};
use crate::judge::Winner;

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub a: String,
    pub b: String,
    pub winner: Winner,
}

/// A connected, balanced ring schedule with unique unordered pairs.
pub fn tournament_pairs(
    candidates: &[String],
    opponents: usize,
    seed: u64,
) -> Result<Vec<(String, String)>> {
    if opponents < 2 || !opponents.is_multiple_of(2) {
        return Err(EvolveError::Ranking(
            "opponents must be even and at least 2".into(),
        ));
    }
    if candidates.is_empty() {
        return Err(EvolveError::Ranking("no candidates".into()));
    }

    let mut names = candidates.to_vec();
    let unique: std::collections::BTreeSet<&String> = names.iter().collect();
    if unique.len() != names.len() {
        return Err(EvolveError::Ranking(
            "candidate names must be unique".into(),
        ));
    }
    names.sort();
    CpRandom::new(seed).shuffle(&mut names);

    let n = names.len();
    let reach = (opponents / 2).min(n.saturating_sub(1));
    let mut pairs = std::collections::BTreeSet::new();
    for i in 0..n {
        for distance in 1..=reach {
            let j = (i + distance) % n;
            if i == j {
                continue;
            }
            let (left, right) = if names[i] < names[j] {
                (names[i].clone(), names[j].clone())
            } else {
                (names[j].clone(), names[i].clone())
            };
            pairs.insert((left, right));
        }
    }

    let mut oriented: Vec<(String, String)> = pairs.into_iter().collect();
    let mut coin = CpRandom::new(seed.wrapping_add(1));
    for pair in &mut oriented {
        if coin.random_f64() < 0.5 {
            std::mem::swap(&mut pair.0, &mut pair.1);
        }
    }
    Ok(oriented)
}

/// Fit Bradley–Terry scores with the MM update and return `ln(score)`.
pub fn bradley_terry(
    episodes: &[String],
    matches: &[Match],
    iterations: usize,
    prior: f64,
) -> Result<BTreeMap<String, f64>> {
    if episodes.is_empty() {
        return Err(EvolveError::Ranking("no episodes".into()));
    }
    if iterations == 0 {
        return Err(EvolveError::Ranking(
            "ranking iterations must be positive".into(),
        ));
    }
    if !prior.is_finite() || prior < 0.0 {
        return Err(EvolveError::Ranking(
            "prior must be finite and non-negative".into(),
        ));
    }

    let mut index = BTreeMap::new();
    for (position, name) in episodes.iter().enumerate() {
        if index.insert(name.clone(), position).is_some() {
            return Err(EvolveError::Ranking(format!(
                "episode {name:?} appears twice"
            )));
        }
    }

    let n = episodes.len();
    let mut wins = vec![vec![0.0_f64; n]; n];
    for pair in matches {
        let a = *index
            .get(&pair.a)
            .ok_or_else(|| EvolveError::Ranking(format!("unknown episode {:?}", pair.a)))?;
        let b = *index
            .get(&pair.b)
            .ok_or_else(|| EvolveError::Ranking(format!("unknown episode {:?}", pair.b)))?;
        if a == b {
            return Err(EvolveError::Ranking(
                "an episode cannot be matched with itself".into(),
            ));
        }
        match pair.winner {
            Winner::A => wins[a][b] += 1.0,
            Winner::B => wins[b][a] += 1.0,
            Winner::Tie => {
                wins[a][b] += 0.5;
                wins[b][a] += 0.5;
            }
        }
    }

    let mut p = vec![1.0_f64; n];
    for _ in 0..iterations {
        let mut next = vec![1.0_f64; n];
        for i in 0..n {
            let mut numerator = prior;
            for value in &wins[i] {
                numerator += value;
            }
            let mut denominator = if prior > 0.0 {
                prior / (p[i] + 1.0)
            } else {
                0.0
            };
            for j in 0..n {
                if i == j {
                    continue;
                }
                let played = wins[i][j] + wins[j][i];
                if played > 0.0 {
                    denominator += played / (p[i] + p[j]);
                }
            }
            next[i] = if denominator > 0.0 {
                numerator / denominator
            } else {
                p[i]
            };
        }
        let mean = next.iter().sum::<f64>() / n as f64;
        if mean.is_finite() && mean > 0.0 {
            for value in &mut next {
                *value /= mean;
            }
        }
        p = next;
    }

    Ok(episodes
        .iter()
        .enumerate()
        .map(|(position, name)| (name.clone(), p[position].ln()))
        .collect())
}

/// CPython-compatible MT19937.
pub struct CpRandom {
    state: [u32; 624],
    index: usize,
}

impl CpRandom {
    pub fn new(seed: u64) -> Self {
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        let high = (seed >> 32) as u32;
        if high != 0 {
            key.push(high);
        }
        Self::from_key(&key)
    }

    fn from_key(key: &[u32]) -> Self {
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let rounds = 624.max(key.len());
        for _ in 0..rounds {
            let previous = random.state[i - 1];
            random.state[i] = (random.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1664525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                random.state[0] = random.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let previous = random.state[i - 1];
            random.state[i] = (random.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1566083941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                random.state[0] = random.state[623];
                i = 1;
            }
        }
        random.state[0] = 0x8000_0000;
        random.index = 624;
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            let previous = self.state[i - 1];
            self.state[i] = 1812433253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    pub fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        self.next_u32() >> (32 - bits)
    }

    pub fn randbelow(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let bits = 32 - n.leading_zeros();
        loop {
            let value = self.getrandbits(bits);
            if value < n {
                return value;
            }
        }
    }

    pub fn random_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.randbelow(i as u32 + 1) as usize;
            values.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_shuffle() {
        let mut names: Vec<String> = (0..8).map(|i| format!("c{i}")).collect();
        CpRandom::new(1234).shuffle(&mut names);
        assert_eq!(names, vec!["c1", "c3", "c2", "c4", "c5", "c6", "c0", "c7"]);
    }

    #[test]
    fn matches_cpython_tournament_prefix() {
        let names: Vec<String> = (0..18).map(|i| format!("c{i}")).collect();
        let pairs = tournament_pairs(&names, 6, 1234).expect("pairs");
        assert_eq!(pairs.len(), 54);
        let expected = vec![
            ("c0", "c1"),
            ("c11", "c0"),
            ("c0", "c17"),
            ("c4", "c0"),
            ("c6", "c0"),
            ("c0", "c8"),
            ("c1", "c11"),
            ("c1", "c17"),
            ("c4", "c1"),
            ("c6", "c1"),
        ];
        let actual: Vec<(&str, &str)> = pairs
            .iter()
            .take(10)
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn matches_bradley_terry_fixture() {
        let names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let matches = vec![
            Match {
                a: "a".into(),
                b: "b".into(),
                winner: Winner::A,
            },
            Match {
                a: "a".into(),
                b: "c".into(),
                winner: Winner::A,
            },
            Match {
                a: "b".into(),
                b: "c".into(),
                winner: Winner::A,
            },
            Match {
                a: "a".into(),
                b: "d".into(),
                winner: Winner::Tie,
            },
            Match {
                a: "b".into(),
                b: "d".into(),
                winner: Winner::A,
            },
        ];
        let scores = bradley_terry(&names, &matches, 500, 0.5).expect("fit");
        assert!((scores["a"] - 0.6134776024336456).abs() < 1e-9);
        assert!((scores["b"] - 0.173778907615032).abs() < 1e-9);
        assert!((scores["c"] - -1.3002450559427374).abs() < 1e-9);
        assert!((scores["d"] - -0.36976065764660265).abs() < 1e-9);
    }

    #[test]
    fn rejects_odd_opponent_counts() {
        let names = vec!["a".to_string(), "b".to_string()];
        assert!(tournament_pairs(&names, 3, 1).is_err());
    }
}
