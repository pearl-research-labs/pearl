use layout_macro::define_layout;

// everything stored as little endian.
define_layout! {
    mod blake3_columns {
        INPUT_STATE_ROW1: 4,
        INPUT_STATE_ROW2: 128,
        INPUT_STATE_ROW3: 4,
        INPUT_STATE_ROW4: 128,

        STATE1_ROW1: 4,
        STATE1_ROW2: 128,
        STATE1_ROW3: 4,
        STATE1_ROW4: 128,

        STATE2_ROW1: 4,
        STATE2_ROW2: 128,
        STATE2_ROW3: 4,
        STATE2_ROW4: 128,

        STATE3_ROW1: 4,
        STATE3_ROW2: 128,
        STATE3_ROW3: 4,
        STATE3_ROW4: 128,
    }
}
