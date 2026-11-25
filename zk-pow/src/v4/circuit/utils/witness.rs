use core::marker::PhantomData;

/// Raw destination pointers for parallel workers that own disjoint rows.
/// The borrow keeps the original column slices inaccessible while writes run.
pub(crate) struct DisjointColumnWriter<'a, T, const N: usize> {
    columns: [*mut T; N],
    num_rows: usize,
    _borrow: PhantomData<&'a mut [T]>,
}

// SAFETY: metadata is immutable and no safe method accesses the pointed-to
// values. Each unsafe write requires exclusive ownership of its destination
// row. Moving values into those cells on another thread requires T: Send.
unsafe impl<T: Send, const N: usize> Send for DisjointColumnWriter<'_, T, N> {}
unsafe impl<T: Send, const N: usize> Sync for DisjointColumnWriter<'_, T, N> {}

impl<'a, T: Copy, const N: usize> DisjointColumnWriter<'a, T, N> {
    pub(crate) fn new(columns: &'a mut [&mut [T]]) -> Self {
        assert_eq!(columns.len(), N);
        let num_rows = columns.first().map_or(0, |column| column.len());
        assert!(columns.iter().all(|column| column.len() == num_rows));
        Self {
            columns: core::array::from_fn(|c| columns[c].as_mut_ptr()),
            num_rows,
            _borrow: PhantomData,
        }
    }

    /// Writes one row without creating references to destination cells.
    ///
    /// # Safety
    /// `row` must be less than the column length. No other access may read or
    /// write any of this row's destination cells while this call runs.
    #[inline]
    pub(crate) unsafe fn write_row(&self, row: usize, values: &[T; N]) {
        debug_assert!(row < self.num_rows);
        for (base, value) in self.columns.iter().zip(values) {
            // SAFETY: the constructor obtained pointers from the live, uniquely
            // borrowed columns; the caller guarantees row bounds and ownership.
            unsafe { base.add(row).write(*value) };
        }
    }
}
