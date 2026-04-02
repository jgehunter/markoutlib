use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Binary search: index of first element in `arr` strictly greater than `val`.
/// Equivalent to Python's bisect.bisect_right.
#[inline]
fn bisect_right(arr: &ArrayView1<i64>, val: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] <= val {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Compute tick-clock future mids for one partition.
///
/// For n > 0: the n-th quote strictly after each trade.
/// For n == 0: the last quote at or before each trade.
/// For n < 0: the |n|-th quote before the last-at-or-before.
///
/// Returns a float64 numpy array with NaN for out-of-bounds lookups.
#[pyfunction]
fn tick_clock_partition<'py>(
    py: Python<'py>,
    trade_timestamps: PyReadonlyArray1<'py, i64>,
    quote_timestamps: PyReadonlyArray1<'py, i64>,
    quote_mids: PyReadonlyArray1<'py, f64>,
    n: i64,
) -> Bound<'py, PyArray1<f64>> {
    let trade_ts = trade_timestamps.as_array();
    let quote_ts = quote_timestamps.as_array();
    let quote_m = quote_mids.as_array();
    let num_quotes = quote_ts.len();

    let result: Vec<f64> = trade_ts
        .iter()
        .map(|&ts| {
            let idx = bisect_right(&quote_ts, ts);

            let target: i64 = if n > 0 {
                idx as i64 + n - 1
            } else if n == 0 {
                idx as i64 - 1
            } else {
                // negative: count backward from last-at-or-before
                (idx as i64 - 1) + n
            };

            if target >= 0 && (target as usize) < num_quotes {
                quote_m[target as usize]
            } else {
                f64::NAN
            }
        })
        .collect();

    result.into_pyarray(py)
}

#[pymodule]
fn _markoutlib_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(tick_clock_partition, m)?)?;
    Ok(())
}
