use pyo3::prelude::*;

#[pymodule]
fn _markoutlib_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
