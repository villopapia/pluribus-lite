//! Binary checkpoint format shared by the solvers.
//!
//! Layout (little-endian): magic `PLCK`, u32 version, string game name,
//! string algorithm, u64 iteration, [u64; 4] RNG state, u32 n_params + f64 params,
//! u64 n_infosets, then for each infoset sorted by key: u64 key, u8 n,
//! n x f64 regret, n x f64 strat_sum. Written to `<path>.tmp` and renamed, so a
//! crash mid-write never corrupts the previous checkpoint.

use crate::table::{Infoset, Table};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"PLCK";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct Checkpoint {
    pub game: String,
    pub algo: String,
    pub iteration: u64,
    pub rng_state: [u64; 4],
    pub params: Vec<f64>,
    pub table: Table,
}

fn w_u32(w: &mut impl Write, x: u32) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_u64(w: &mut impl Write, x: u64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_f64(w: &mut impl Write, x: f64) -> io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}
fn r_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn r_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn r_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn r_str(r: &mut impl Read) -> io::Result<String> {
    let n = r_u32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

impl Checkpoint {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)?;
            }
        }
        let tmp = path.with_extension("tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            w.write_all(MAGIC)?;
            w_u32(&mut w, VERSION)?;
            w_str(&mut w, &self.game)?;
            w_str(&mut w, &self.algo)?;
            w_u64(&mut w, self.iteration)?;
            for x in self.rng_state {
                w_u64(&mut w, x)?;
            }
            w_u32(&mut w, self.params.len() as u32)?;
            for &p in &self.params {
                w_f64(&mut w, p)?;
            }
            let mut keys: Vec<u64> = self.table.map.keys().copied().collect();
            keys.sort_unstable();
            w_u64(&mut w, keys.len() as u64)?;
            for k in keys {
                let inf = &self.table.map[&k];
                w_u64(&mut w, k)?;
                w.write_all(&[inf.regret.len() as u8])?;
                for &x in &inf.regret {
                    w_f64(&mut w, x)?;
                }
                for &x in &inf.strat_sum {
                    w_f64(&mut w, x)?;
                }
            }
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        fs::rename(&tmp, path)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad checkpoint magic"));
        }
        let v = r_u32(&mut r)?;
        if v != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("version {v}")));
        }
        let game = r_str(&mut r)?;
        let algo = r_str(&mut r)?;
        let iteration = r_u64(&mut r)?;
        let mut rng_state = [0u64; 4];
        for x in rng_state.iter_mut() {
            *x = r_u64(&mut r)?;
        }
        let np = r_u32(&mut r)? as usize;
        let mut params = Vec::with_capacity(np);
        for _ in 0..np {
            params.push(r_f64(&mut r)?);
        }
        let n_inf = r_u64(&mut r)? as usize;
        let mut table = Table::new();
        for _ in 0..n_inf {
            let k = r_u64(&mut r)?;
            let mut nb = [0u8; 1];
            r.read_exact(&mut nb)?;
            let n = nb[0] as usize;
            let mut inf = Infoset::new(n);
            for x in inf.regret.iter_mut() {
                *x = r_f64(&mut r)?;
            }
            for x in inf.strat_sum.iter_mut() {
                *x = r_f64(&mut r)?;
            }
            table.map.insert(k, inf);
        }
        Ok(Checkpoint { game, algo, iteration, rng_state, params, table })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut table = Table::new();
        table.get_or_insert(5, 2).regret = vec![1.5, -2.25];
        table.get_or_insert(9, 3).strat_sum = vec![0.1, 0.2, 0.7];
        let c = Checkpoint {
            game: "kuhn".into(),
            algo: "mccfr".into(),
            iteration: 1234,
            rng_state: [1, 2, 3, 4],
            params: vec![0.5, -3.0],
            table,
        };
        let dir = std::env::temp_dir().join(format!("plck_test_{}", std::process::id()));
        let path = dir.join("rt.ckpt");
        c.save(&path).unwrap();
        let d = Checkpoint::load(&path).unwrap();
        assert_eq!(c, d);
        fs::remove_dir_all(dir).ok();
    }
}
