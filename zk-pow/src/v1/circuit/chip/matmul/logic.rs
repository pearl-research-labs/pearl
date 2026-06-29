use crate::v1::circuit::chip::blake3::program::MatDwordId;

#[derive(Clone, Debug, Copy, Default)]
pub struct MatmulLogic {
    pub is_reset_cumsum: bool, // If true, disregard previous row -- as if in first row

    // See also pearl_trace.rs::read_tile:
    // Top left corner dword of A to load. if its index a_i then we load this submatrix of A:
    // a_i    a_i+2
    // a_i+1  a_i+3
    pub a_dword: Option<MatDwordId>,
    // Top left corner dword of B^t to load. if its index b_i then we load this submatrix of B^t:
    // b_i    b_i+2
    // b_i+1  b_i+3
    pub b_dword: Option<MatDwordId>,
}

impl MatmulLogic {
    pub fn is_update_cumsum(&self) -> bool {
        debug_assert_eq!(self.a_dword.is_some(), self.b_dword.is_some());
        self.a_dword.is_some()
    }
}
