use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use fnv::FnvHashMap;

/// Rust implementation of UMI grouping for performance-critical operations
#[pyfunction]
fn group_umis_fast(reads: Vec<(String, String, String, String, String)>) -> PyResult<Vec<Vec<usize>>> {
    // reads: (chrom, pos, ref, umi, allele)
    // Returns: groups of read indices sharing the same UMI at the same locus
    
    let mut umi_groups: FnvHashMap<String, Vec<usize>> = FnvHashMap::default();
    
    for (idx, (chrom, pos, ref_allele, umi, _allele)) in reads.iter().enumerate() {
        let site_key = format!("{}:{}:{}", chrom, pos, ref_allele);
        let group_key = format!("{}:{}", site_key, umi);
        
        umi_groups.entry(group_key).or_insert_with(Vec::new).push(idx);
    }
    
    Ok(umi_groups.into_values().collect())
}

/// Fast consensus calling for UMI families
#[pyfunction] 
fn call_consensus_fast(
    alleles: Vec<String>, 
    qualities: Vec<i32>,
    min_quality: i32,
    consensus_threshold: f64
) -> PyResult<Option<String>> {
    
    // Filter by quality
    let quality_reads: Vec<(String, i32)> = alleles
        .into_iter()
        .zip(qualities.into_iter())
        .filter(|(_, q)| *q >= min_quality)
        .collect();
    
    if quality_reads.is_empty() {
        return Ok(None);
    }
    
    // Count alleles weighted by quality
    let mut allele_weights: HashMap<String, f64> = HashMap::new();
    let mut total_weight = 0.0;
    
    for (allele, quality) in quality_reads {
        let weight = 10.0_f64.powf(quality as f64 / 10.0);
        *allele_weights.entry(allele).or_insert(0.0) += weight;
        total_weight += weight;
    }
    
    if total_weight == 0.0 {
        return Ok(None);
    }
    
    // Find consensus allele
    let best_allele = allele_weights
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(allele, _)| allele);
    
    if let Some(allele) = best_allele {
        let consensus_fraction = allele_weights[allele] / total_weight;
        if consensus_fraction >= consensus_threshold {
            Ok(Some(allele.clone()))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

/// Calculate edit distance between two UMI sequences
#[pyfunction]
fn umi_edit_distance(umi1: &str, umi2: &str) -> PyResult<u32> {
    if umi1.len() != umi2.len() {
        return Ok(std::cmp::max(umi1.len(), umi2.len()) as u32);
    }
    
    let distance = umi1
        .chars()
        .zip(umi2.chars())
        .map(|(c1, c2)| if c1 != c2 { 1 } else { 0 })
        .sum();
    
    Ok(distance)
}

/// Parallel processing of multiple UMI families
#[pyfunction]
fn process_families_parallel(
    families_data: Vec<(Vec<String>, Vec<i32>)>, // (alleles, qualities) for each family
    min_quality: i32,
    consensus_threshold: f64
) -> PyResult<Vec<Option<String>>> {
    
    let results: Vec<Option<String>> = families_data
        .into_par_iter()
        .map(|(alleles, qualities)| {
            call_consensus_fast(alleles, qualities, min_quality, consensus_threshold)
                .unwrap_or(None)
        })
        .collect();
    
    Ok(results)
}

/// Benchmark function for performance testing
#[pyfunction]
fn benchmark_umi_processing(n_reads: usize, n_families: usize) -> PyResult<f64> {
    use std::time::Instant;
    
    // Generate synthetic test data
    let mut reads = Vec::with_capacity(n_reads);
    for i in 0..n_reads {
        let family_id = i % n_families;
        reads.push((
            "chr1".to_string(),
            "1000000".to_string(), 
            "A".to_string(),
            format!("UMI{:06}", family_id),
            if i % 10 == 0 { "T".to_string() } else { "A".to_string() }
        ));
    }
    
    let start = Instant::now();
    let _groups = group_umis_fast(reads)?;
    let duration = start.elapsed();
    
    Ok(duration.as_secs_f64())
}

/// Python module definition
#[pymodule]
fn precise_mrd_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(group_umis_fast, m)?)?;
    m.add_function(wrap_pyfunction!(call_consensus_fast, m)?)?;
    m.add_function(wrap_pyfunction!(umi_edit_distance, m)?)?;
    m.add_function(wrap_pyfunction!(process_families_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_umi_processing, m)?)?;
    Ok(())
}