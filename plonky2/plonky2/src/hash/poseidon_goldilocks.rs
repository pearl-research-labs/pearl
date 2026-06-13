//! Implementations for Poseidon over Goldilocks field of widths 8 and 12.
//!
//! These contents of the implementations *must* be generated using the
//! `poseidon_constants.sage` script in the `0xPolygonZero/hash-constants` repository
//! (https://github.com/0xPolygonZero/hash-constants).

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
use plonky2_field::types::Field;

use crate::field::goldilocks_field::GoldilocksField;
use crate::hash::poseidon::{Poseidon, N_PARTIAL_ROUNDS};

#[rustfmt::skip]
impl Poseidon for GoldilocksField {
    // The MDS matrix we use is C + D, where C is the circulant matrix whose first row is given by
    // `MDS_MATRIX_CIRC`, and D is the diagonal matrix whose diagonal is given by `MDS_MATRIX_DIAG`.
    //
    // WARNING: If the MDS matrix is changed, then the following
    // constants need to be updated accordingly:
    //  - FAST_PARTIAL_ROUND_CONSTANTS
    //  - FAST_PARTIAL_ROUND_VS
    //  - FAST_PARTIAL_ROUND_W_HATS
    //  - FAST_PARTIAL_ROUND_INITIAL_MATRIX
    const MDS_MATRIX_CIRC: [u64; 12] = [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20];
    const MDS_MATRIX_DIAG: [u64; 12] = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; 12]  = [
        0x3cc3f892184df408, 0xe993fd841e7e97f1, 0xf2831d3575f0f3af, 0xd2500e0a350994ca,
        0xc5571f35d7288633, 0x91d89c5184109a02, 0xf37f925d04e5667b, 0x2d6e448371955a69,
        0x740ef19ce01398a1, 0x694d24c0752fdf45, 0x60936af96ee2f148, 0xc33448feadc78f0c,
    ];

    const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS]  = [
        0x74cb2e819ae421ab, 0xd2559d2370e7f663, 0x62bf78acf843d17c, 0xd5ab7b67e14d1fb4,
        0xb9fe2ae6e0969bdc, 0xe33fdf79f92a10e8, 0x0ea2bb4c2b25989b, 0xca9121fbf9d38f06,
        0xbdd9b0aa81f58fa4, 0x83079fa4ecf20d7e, 0x650b838edfcc4ad3, 0x77180c88583c76ac,
        0xaf8c20753143a180, 0xb8ccfe9989a39175, 0x954a1729f60cc9c5, 0xdeb5b550c4dca53b,
        0xf01bb0b00f77011e, 0xa1ebb404b676afd9, 0x860b6e1597a0173e, 0x308bb65a036acbce,
        0x1aca78f31c97c876, 0x0,
    ];

    const FAST_PARTIAL_ROUND_VS: [[u64; 12 - 1]; N_PARTIAL_ROUNDS] = [
        [0x94877900674181c3, 0xc6c67cc37a2a2bbd, 0xd667c2055387940f, 0x0ba63a63e94b5ff0,
         0x99460cc41b8f079f, 0x7ff02375ed524bb3, 0xea0870b47a8caf0e, 0xabcad82633b7bc9d,
         0x3b8d135261052241, 0xfb4515f5e5b0d539, 0x3ee8011c2b37f77c, ],
        [0x0adef3740e71c726, 0xa37bf67c6f986559, 0xc6b16f7ed4fa1b00, 0x6a065da88d8bfc3c,
         0x4cabc0916844b46f, 0x407faac0f02e78d1, 0x07a786d9cf0852cf, 0x42433fb6949a629a,
         0x891682a147ce43b0, 0x26cfd58e7b003b55, 0x2bbf0ed7b657acb3, ],
        [0x481ac7746b159c67, 0xe367de32f108e278, 0x73f260087ad28bec, 0x5cfc82216bc1bdca,
         0xcaccc870a2663a0e, 0xdb69cd7b4298c45d, 0x7bc9e0c57243e62d, 0x3cc51c5d368693ae,
         0x366b4e8cc068895b, 0x2bd18715cdabbca4, 0xa752061c4f33b8cf, ],
        [0xb22d2432b72d5098, 0x9e18a487f44d2fe4, 0x4b39e14ce22abd3c, 0x9e77fde2eb315e0d,
         0xca5e0385fe67014d, 0x0c2cb99bf1b6bddb, 0x99ec1cd2a4460bfe, 0x8577a815a2ff843f,
         0x7d80a6b4fd6518a5, 0xeb6c67123eab62cb, 0x8f7851650eca21a5, ],
        [0x11ba9a1b81718c2a, 0x9f7d798a3323410c, 0xa821855c8c1cf5e5, 0x535e8d6fac0031b2,
         0x404e7c751b634320, 0xa729353f6e55d354, 0x4db97d92e58bb831, 0xb53926c27897bf7d,
         0x965040d52fe115c5, 0x9565fa41ebd31fd7, 0xaae4438c877ea8f4, ],
        [0x37f4e36af6073c6e, 0x4edc0918210800e9, 0xc44998e99eae4188, 0x9f4310d05d068338,
         0x9ec7fe4350680f29, 0xc5b2c1fdc0b50874, 0xa01920c5ef8b2ebe, 0x59fa6f8bd91d58ba,
         0x8bfc9eb89b515a82, 0xbe86a7a2555ae775, 0xcbb8bbaa3810babf, ],
        [0x577f9a9e7ee3f9c2, 0x88c522b949ace7b1, 0x82f07007c8b72106, 0x8283d37c6675b50e,
         0x98b074d9bbac1123, 0x75c56fb7758317c1, 0xfed24e206052bc72, 0x26d7c3d1bc07dae5,
         0xf88c5e441e28dbb4, 0x4fe27f9f96615270, 0x514d4ba49c2b14fe, ],
        [0xf02a3ac068ee110b, 0x0a3630dafb8ae2d7, 0xce0dc874eaf9b55c, 0x9a95f6cff5b55c7e,
         0x626d76abfed00c7b, 0xa0c1cf1251c204ad, 0xdaebd3006321052c, 0x3d4bd48b625a8065,
         0x7f1e584e071f6ed2, 0x720574f0501caed3, 0xe3260ba93d23540a, ],
        [0xab1cbd41d8c1e335, 0x9322ed4c0bc2df01, 0x51c3c0983d4284e5, 0x94178e291145c231,
         0xfd0f1a973d6b2085, 0xd427ad96e2b39719, 0x8a52437fecaac06b, 0xdc20ee4b8c4c9a80,
         0xa2c98e9549da2100, 0x1603fe12613db5b6, 0x0e174929433c5505, ],
        [0x3d4eab2b8ef5f796, 0xcfff421583896e22, 0x4143cb32d39ac3d9, 0x22365051b78a5b65,
         0x6f7fd010d027c9b6, 0xd9dd36fba77522ab, 0xa44cf1cb33e37165, 0x3fc83d3038c86417,
         0xc4588d418e88d270, 0xce1320f10ab80fe2, 0xdb5eadbbec18de5d, ],
        [0x1183dfce7c454afd, 0x21cea4aa3d3ed949, 0x0fce6f70303f2304, 0x19557d34b55551be,
         0x4c56f689afc5bbc9, 0xa1e920844334f944, 0xbad66d423d2ec861, 0xf318c785dc9e0479,
         0x99e2032e765ddd81, 0x400ccc9906d66f45, 0xe1197454db2e0dd9, ],
        [0x84d1ecc4d53d2ff1, 0xd8af8b9ceb4e11b6, 0x335856bb527b52f4, 0xc756f17fb59be595,
         0xc0654e4ea5553a78, 0x9e9a46b61f2ea942, 0x14fc8b5b3b809127, 0xd7009f0f103be413,
         0x3e0ee7b7a9fb4601, 0xa74e888922085ed7, 0xe80a7cde3d4ac526, ],
        [0x238aa6daa612186d, 0x9137a5c630bad4b4, 0xc7db3817870c5eda, 0x217e4f04e5718dc9,
         0xcae814e2817bd99d, 0xe3292e7ab770a8ba, 0x7bb36ef70b6b9482, 0x3c7835fb85bca2d3,
         0xfe2cdf8ee3c25e86, 0x61b3915ad7274b20, 0xeab75ca7c918e4ef, ],
        [0xd6e15ffc055e154e, 0xec67881f381a32bf, 0xfbb1196092bf409c, 0xdc9d2e07830ba226,
         0x0698ef3245ff7988, 0x194fae2974f8b576, 0x7a5d9bea6ca4910e, 0x7aebfea95ccdd1c9,
         0xf9bd38a67d5f0e86, 0xfa65539de65492d8, 0xf0dfcbe7653ff787, ],
        [0x0bd87ad390420258, 0x0ad8617bca9e33c8, 0x0c00ad377a1e2666, 0x0ac6fc58b3f0518f,
         0x0c0cc8a892cc4173, 0x0c210accb117bc21, 0x0b73630dbb46ca18, 0x0c8be4920cbd4a54,
         0x0bfe877a21be1690, 0x0ae790559b0ded81, 0x0bf50db2f8d6ce31, ],
        [0x000cf29427ff7c58, 0x000bd9b3cf49eec8, 0x000d1dc8aa81fb26, 0x000bc792d5c394ef,
         0x000d2ae0b2266453, 0x000d413f12c496c1, 0x000c84128cfed618, 0x000db5ebd48fc0d4,
         0x000d1b77326dcb90, 0x000beb0ccc145421, 0x000d10e5b22b11d1, ],
        [0x00000e24c99adad8, 0x00000cf389ed4bc8, 0x00000e580cbf6966, 0x00000cde5fd7e04f,
         0x00000e63628041b3, 0x00000e7e81a87361, 0x00000dabe78f6d98, 0x00000efb14cac554,
         0x00000e5574743b10, 0x00000d05709f42c1, 0x00000e4690c96af1, ],
        [0x0000000f7157bc98, 0x0000000e3006d948, 0x0000000fa65811e6, 0x0000000e0d127e2f,
         0x0000000fc18bfe53, 0x0000000fd002d901, 0x0000000eed6461d8, 0x0000001068562754,
         0x0000000fa0236f50, 0x0000000e3af13ee1, 0x0000000fa460f6d1, ],
        [0x0000000011131738, 0x000000000f56d588, 0x0000000011050f86, 0x000000000f848f4f,
         0x00000000111527d3, 0x00000000114369a1, 0x00000000106f2f38, 0x0000000011e2ca94,
         0x00000000110a29f0, 0x000000000fa9f5c1, 0x0000000010f625d1, ],
        [0x000000000011f718, 0x000000000010b6c8, 0x0000000000134a96, 0x000000000010cf7f,
         0x0000000000124d03, 0x000000000013f8a1, 0x0000000000117c58, 0x0000000000132c94,
         0x0000000000134fc0, 0x000000000010a091, 0x0000000000128961, ],
        [0x0000000000001300, 0x0000000000001750, 0x000000000000114e, 0x000000000000131f,
         0x000000000000167b, 0x0000000000001371, 0x0000000000001230, 0x000000000000182c,
         0x0000000000001368, 0x0000000000000f31, 0x00000000000015c9, ],
        [0x0000000000000014, 0x0000000000000022, 0x0000000000000012, 0x0000000000000027,
         0x000000000000000d, 0x000000000000000d, 0x000000000000001c, 0x0000000000000002,
         0x0000000000000010, 0x0000000000000029, 0x000000000000000f, ],
    ];

    const FAST_PARTIAL_ROUND_W_HATS: [[u64; 12 - 1]; N_PARTIAL_ROUNDS] = [
        [0x3d999c961b7c63b0, 0x814e82efcd172529, 0x2421e5d236704588, 0x887af7d4dd482328,
         0xa5e9c291f6119b27, 0xbdc52b2676a4b4aa, 0x64832009d29bcf57, 0x09c4155174a552cc,
         0x463f9ee03d290810, 0xc810936e64982542, 0x043b1c289f7bc3ac, ],
        [0x673655aae8be5a8b, 0xd510fe714f39fa10, 0x2c68a099b51c9e73, 0xa667bfa9aa96999d,
         0x4d67e72f063e2108, 0xf84dde3e6acda179, 0x40f9cc8c08f80981, 0x5ead032050097142,
         0x6591b02092d671bb, 0x00e18c71963dd1b7, 0x8a21bcd24a14218a, ],
        [0x202800f4addbdc87, 0xe4b5bdb1cc3504ff, 0xbe32b32a825596e7, 0x8e0f68c5dc223b9a,
         0x58022d9e1c256ce3, 0x584d29227aa073ac, 0x8b9352ad04bef9e7, 0xaead42a3f445ecbf,
         0x3c667a1d833a3cca, 0xda6f61838efa1ffe, 0xe8f749470bd7c446, ],
        [0xc5b85bab9e5b3869, 0x45245258aec51cf7, 0x16e6b8e68b931830, 0xe2ae0f051418112c,
         0x0470e26a0093a65b, 0x6bef71973a8146ed, 0x119265be51812daf, 0xb0be7356254bea2e,
         0x8584defff7589bd7, 0x3c5fe4aeb1fb52ba, 0x9e7cd88acf543a5e, ],
        [0x179be4bba87f0a8c, 0xacf63d95d8887355, 0x6696670196b0074f, 0xd99ddf1fe75085f9,
         0xc2597881fef0283b, 0xcf48395ee6c54f14, 0x15226a8e4cd8d3b6, 0xc053297389af5d3b,
         0x2c08893f0d1580e2, 0x0ed3cbcff6fcc5ba, 0xc82f510ecf81f6d0, ],
        [0x94b06183acb715cc, 0x500392ed0d431137, 0x861cc95ad5c86323, 0x05830a443f86c4ac,
         0x3b68225874a20a7c, 0x10b3309838e236fb, 0x9b77fc8bcd559e2c, 0xbdecf5e0cb9cb213,
         0x30276f1221ace5fa, 0x7935dd342764a144, 0xeac6db520bb03708, ],
        [0x7186a80551025f8f, 0x622247557e9b5371, 0xc4cbe326d1ad9742, 0x55f1523ac6a23ea2,
         0xa13dfe77a3d52f53, 0xe30750b6301c0452, 0x08bd488070a3a32b, 0xcd800caef5b72ae3,
         0x83329c90f04233ce, 0xb5b99e6664a0a3ee, 0x6b0731849e200a7f, ],
        [0xec3fabc192b01799, 0x382b38cee8ee5375, 0x3bfb6c3f0e616572, 0x514abd0cf6c7bc86,
         0x47521b1361dcc546, 0x178093843f863d14, 0xad1003c5d28918e7, 0x738450e42495bc81,
         0xaf947c59af5e4047, 0x4653fb0685084ef2, 0x057fde2062ae35bf, ],
        [0xe376678d843ce55e, 0x66f3860d7514e7fc, 0x7817f3dfff8b4ffa, 0x3929624a9def725b,
         0x0126ca37f215a80a, 0xfce2f5d02762a303, 0x1bc927375febbad7, 0x85b481e5243f60bf,
         0x2d3c5f42a39c91a0, 0x0811719919351ae8, 0xf669de0add993131, ],
        [0x7de38bae084da92d, 0x5b848442237e8a9b, 0xf6c705da84d57310, 0x31e6a4bdb6a49017,
         0x889489706e5c5c0f, 0x0e4a205459692a1b, 0xbac3fa75ee26f299, 0x5f5894f4057d755e,
         0xb0dc3ecd724bb076, 0x5e34d8554a6452ba, 0x04f78fd8c1fdcc5f, ],
        [0x4dd19c38779512ea, 0xdb79ba02704620e9, 0x92a29a3675a5d2be, 0xd5177029fe495166,
         0xd32b3298a13330c1, 0x251c4a3eb2c5f8fd, 0xe1c48b26e0d98825, 0x3301d3362a4ffccb,
         0x09bb6c88de8cd178, 0xdc05b676564f538a, 0x60192d883e473fee, ],
        [0x16b9774801ac44a0, 0x3cb8411e786d3c8e, 0xa86e9cf505072491, 0x0178928152e109ae,
         0x5317b905a6e1ab7b, 0xda20b3be7f53d59f, 0xcb97dedecebee9ad, 0x4bd545218c59f58d,
         0x77dc8d856c05a44a, 0x87948589e4f243fd, 0x7e5217af969952c2, ],
        [0xbc58987d06a84e4d, 0x0b5d420244c9cae3, 0xa3c4711b938c02c0, 0x3aace640a3e03990,
         0x865a0f3249aacd8a, 0x8d00b2a7dbed06c7, 0x6eacb905beb7e2f8, 0x045322b216ec3ec7,
         0xeb9de00d594828e6, 0x088c5f20df9e5c26, 0xf555f4112b19781f, ],
        [0xa8cedbff1813d3a7, 0x50dcaee0fd27d164, 0xf1cb02417e23bd82, 0xfaf322786e2abe8b,
         0x937a4315beb5d9b6, 0x1b18992921a11d85, 0x7d66c4368b3c497b, 0x0e7946317a6b4e99,
         0xbe4430134182978b, 0x3771e82493ab262d, 0xa671690d8095ce82, ],
        [0xb035585f6e929d9d, 0xba1579c7e219b954, 0xcb201cf846db4ba3, 0x287bf9177372cf45,
         0xa350e4f61147d0a6, 0xd5d0ecfb50bcff99, 0x2e166aa6c776ed21, 0xe1e66c991990e282,
         0x662b329b01e7bb38, 0x8aa674b36144d9a9, 0xcbabf78f97f95e65, ],
        [0xeec24b15a06b53fe, 0xc8a7aa07c5633533, 0xefe9c6fa4311ad51, 0xb9173f13977109a1,
         0x69ce43c9cc94aedc, 0xecf623c9cd118815, 0x28625def198c33c7, 0xccfc5f7de5c3636a,
         0xf5e6c40f1621c299, 0xcec0e58c34cb64b1, 0xa868ea113387939f, ],
        [0xd8dddbdc5ce4ef45, 0xacfc51de8131458c, 0x146bb3c0fe499ac0, 0x9e65309f15943903,
         0x80d0ad980773aa70, 0xf97817d4ddbf0607, 0xe4626620a75ba276, 0x0dfdc7fd6fc74f66,
         0xf464864ad6f2bb93, 0x02d55e52a5d44414, 0xdd8de62487c40925, ],
        [0xc15acf44759545a3, 0xcbfdcf39869719d4, 0x33f62042e2f80225, 0x2599c5ead81d8fa3,
         0x0b306cb6c1d7c8d0, 0x658c80d3df3729b1, 0xe8d1b2b21b41429c, 0xa1b67f09d4b3ccb8,
         0x0e1adf8b84437180, 0x0d593a5e584af47b, 0xa023d94c56e151c7, ],
        [0x49026cc3a4afc5a6, 0xe06dff00ab25b91b, 0x0ab38c561e8850ff, 0x92c3c8275e105eeb,
         0xb65256e546889bd0, 0x3c0468236ea142f6, 0xee61766b889e18f2, 0xa206f41b12c30415,
         0x02fe9d756c9f12d1, 0xe9633210630cbf12, 0x1ffea9fe85a0b0b1, ],
        [0x81d1ae8cc50240f3, 0xf4c77a079a4607d7, 0xed446b2315e3efc1, 0x0b0a6b70915178c3,
         0xb11ff3e089f15d9a, 0x1d4dba0b7ae9cc18, 0x65d74e2f43b48d05, 0xa2df8c6b8ae0804a,
         0xa4e6f0a8c33348a6, 0xc0a26efc7be5669b, 0xa6b6582c547d0d60, ],
        [0x84afc741f1c13213, 0x2f8f43734fc906f3, 0xde682d72da0a02d9, 0x0bb005236adb9ef2,
         0x5bdf35c10a8b5624, 0x0739a8a343950010, 0x52f515f44785cfbc, 0xcbaf4e5d82856c60,
         0xac9ea09074e3e150, 0x8f0fa011a2035fb0, 0x1a37905d8450904a, ],
        [0x3abeb80def61cc85, 0x9d19c9dd4eac4133, 0x075a652d9641a985, 0x9daf69ae1b67e667,
         0x364f71da77920a18, 0x50bd769f745c95b1, 0xf223d1180dbbf3fc, 0x2f885e584e04aa99,
         0xb69a0fa70aea684a, 0x09584acaa6e062a0, 0x0bc051640145b19b, ],
    ];

    // NB: This is in ROW-major order to support cache-friendly pre-multiplication.
    const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; 12 - 1]; 12 - 1] = [
        [0x80772dc2645b280b, 0xdc927721da922cf8, 0xc1978156516879ad, 0x90e80c591f48b603,
         0x3a2432625475e3ae, 0x00a2d4321cca94fe, 0x77736f524010c932, 0x904d3f2804a36c54,
         0xbf9b39e28a16f354, 0x3a1ded54a6cd058b, 0x42392870da5737cf, ],
        [0xe796d293a47a64cb, 0xb124c33152a2421a, 0x0ee5dc0ce131268a, 0xa9032a52f930fae6,
         0x7e33ca8c814280de, 0xad11180f69a8c29e, 0xc75ac6d5b5a10ff3, 0xf0674a8dc5a387ec,
         0xb36d43120eaa5e2b, 0x6f232aab4b533a25, 0x3a1ded54a6cd058b, ],
        [0xdcedab70f40718ba, 0x14a4a64da0b2668f, 0x4715b8e5ab34653b, 0x1e8916a99c93a88e,
         0xbba4b5d86b9a3b2c, 0xe76649f9bd5d5c2e, 0xaf8e2518a1ece54d, 0xdcda1344cdca873f,
         0xcd080204256088e5, 0xb36d43120eaa5e2b, 0xbf9b39e28a16f354, ],
        [0xf4a437f2888ae909, 0xc537d44dc2875403, 0x7f68007619fd8ba9, 0xa4911db6a32612da,
         0x2f7e9aade3fdaec1, 0xe7ffd578da4ea43d, 0x43a608e7afa6b5c2, 0xca46546aa99e1575,
         0xdcda1344cdca873f, 0xf0674a8dc5a387ec, 0x904d3f2804a36c54, ],
        [0xf97abba0dffb6c50, 0x5e40f0c9bb82aab5, 0x5996a80497e24a6b, 0x07084430a7307c9a,
         0xad2f570a5b8545aa, 0xab7f81fef4274770, 0xcb81f535cf98c9e9, 0x43a608e7afa6b5c2,
         0xaf8e2518a1ece54d, 0xc75ac6d5b5a10ff3, 0x77736f524010c932, ],
        [0x7f8e41e0b0a6cdff, 0x4b1ba8d40afca97d, 0x623708f28fca70e8, 0xbf150dc4914d380f,
         0xc26a083554767106, 0x753b8b1126665c22, 0xab7f81fef4274770, 0xe7ffd578da4ea43d,
         0xe76649f9bd5d5c2e, 0xad11180f69a8c29e, 0x00a2d4321cca94fe, ],
        [0x726af914971c1374, 0x1d7f8a2cce1a9d00, 0x18737784700c75cd, 0x7fb45d605dd82838,
         0x862361aeab0f9b6e, 0xc26a083554767106, 0xad2f570a5b8545aa, 0x2f7e9aade3fdaec1,
         0xbba4b5d86b9a3b2c, 0x7e33ca8c814280de, 0x3a2432625475e3ae, ],
        [0x64dd936da878404d, 0x4db9a2ead2bd7262, 0xbe2e19f6d07f1a83, 0x02290fe23c20351a,
         0x7fb45d605dd82838, 0xbf150dc4914d380f, 0x07084430a7307c9a, 0xa4911db6a32612da,
         0x1e8916a99c93a88e, 0xa9032a52f930fae6, 0x90e80c591f48b603, ],
        [0x85418a9fef8a9890, 0xd8a2eb7ef5e707ad, 0xbfe85ababed2d882, 0xbe2e19f6d07f1a83,
         0x18737784700c75cd, 0x623708f28fca70e8, 0x5996a80497e24a6b, 0x7f68007619fd8ba9,
         0x4715b8e5ab34653b, 0x0ee5dc0ce131268a, 0xc1978156516879ad, ],
        [0x156048ee7a738154, 0x91f7562377e81df5, 0xd8a2eb7ef5e707ad, 0x4db9a2ead2bd7262,
         0x1d7f8a2cce1a9d00, 0x4b1ba8d40afca97d, 0x5e40f0c9bb82aab5, 0xc537d44dc2875403,
         0x14a4a64da0b2668f, 0xb124c33152a2421a, 0xdc927721da922cf8, ],
        [0xd841e8ef9dde8ba0, 0x156048ee7a738154, 0x85418a9fef8a9890, 0x64dd936da878404d,
         0x726af914971c1374, 0x7f8e41e0b0a6cdff, 0xf97abba0dffb6c50, 0xf4a437f2888ae909,
         0xdcedab70f40718ba, 0xe796d293a47a64cb, 0x80772dc2645b280b, ],
    ];

    /// AVX-512 mds_layer: same algorithm but compiled with AVX-512 enabled
    /// so the compiler can use zmm registers for the i64 arithmetic.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline(always)]
    fn mds_layer(state: &[Self; 12]) -> [Self; 12] {
        let mut result = [GoldilocksField::ZERO; 12];

        // Using the linearity of the operations we can split the state into a low||high decomposition
        // and operate on each with no overflow and then combine/reduce the result to a field element.
        let mut state_l = [0u64; 12];
        let mut state_h = [0u64; 12];

        for r in 0..12 {
            let s = state[r].0;
            state_h[r] = s >> 32;
            state_l[r] = (s as u32) as u64;
        }

        let state_h = poseidon12_mds::mds_multiply_freq(state_h);
        let state_l = poseidon12_mds::mds_multiply_freq(state_l);

        for r in 0..12 {
            let s = state_l[r] as u128 + ((state_h[r] as u128) << 32);

            result[r] = GoldilocksField::from_noncanonical_u96((s as u64, (s >> 64) as u32));
        }

        // Add first element with the only non-zero diagonal matrix coefficient.
        let s = Self::MDS_MATRIX_DIAG[0] as u128 * (state[0].0 as u128);
        result[0] += GoldilocksField::from_noncanonical_u96((s as u64, (s >> 64) as u32));

        result
    }

    /// Override poseidon using AVX-512 packed sbox for speed.
    /// The sbox (x^7) is 20% of permutation time. AVX-512 processes 8 field
    /// multiplications in parallel, giving ~2x sbox speedup.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline]
    fn poseidon(input: [Self; 12]) -> [Self; 12] {
        let mut state = input;
        let mut round_ctr = 0;

        // First 4 full rounds (uses overrides below)
        Self::full_rounds(&mut state, &mut round_ctr);

        // 22 partial rounds (uses AVX-512 override below)
        Self::partial_rounds(&mut state, &mut round_ctr);

        // Last 4 full rounds
        Self::full_rounds(&mut state, &mut round_ctr);

        state
    }

    /// AVX-512 optimized partial rounds: fuses partial_first_constant_layer,
    /// mds_partial_layer_init, and the 22 sparse MDS rounds with packed ops.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline(always)]
    fn partial_rounds(state: &mut [Self; 12], round_ctr: &mut usize) {
        use crate::hash::poseidon::{Poseidon, N_PARTIAL_ROUNDS};
        use crate::field::types::{Field, Field64};
        use plonky2_field::ops::Square;
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
        use plonky2_field::packed::PackedField;

        // partial_first_constant_layer: add constants to all 12 elements
        for i in 0..12 {
            let c = GoldilocksField::from_canonical_u64(Self::FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
            state[i] += c;
        }

        // mds_partial_layer_init: AVX-512 optimized 11x11 matrix-vector multiply
        // result[0] = state[0], result[c] = Σ_{r=1}^{11} state[r] * matrix[r-1][c-1]
        // Process columns 1..8 with AVX-512, 9..11 scalar
        let mut result = [Self::ZERO; 12];
        result[0] = state[0];

        // Accumulate into packed result for columns 1..8
        let mut acc = P::ZEROS;
        for r in 1..12 {
            // Broadcast state[r] to all 8 lanes
            let sr = P([state[r]; 8]);
            // Pack matrix[r-1][0..8] (columns 1-8)
            let mut mat_row = P::ZEROS;
            for c in 0..8 {
                mat_row.0[c] = GoldilocksField::from_canonical_u64(
                    Self::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c]
                );
            }
            // Multiply-accumulate: acc += state[r] * matrix_row
            acc = acc + sr * mat_row;
        }
        // Store packed result
        for c in 0..8 {
            result[c + 1] = acc.0[c];
        }

        // Scalar for columns 9..11
        for c in 8..11 {
            for r in 1..12 {
                let t = Self::from_canonical_u64(
                    Self::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c]
                );
                result[c + 1] += state[r] * t;
            }
        }
        *state = result;

        // 22 sparse partial rounds (uses AVX-512 mds_partial_layer_fast override)
        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = Self::sbox_monomial(state[0]);
            unsafe {
                state[0] = state[0].add_canonical_u64(Self::FAST_PARTIAL_ROUND_CONSTANTS[i]);
            }
            *state = Self::mds_partial_layer_fast(state, i);
        }
        *round_ctr += N_PARTIAL_ROUNDS;
    }

    /// AVX-512 fused constant+sbox+mds for full rounds.
    /// Fuses constant_layer + sbox_layer into a single pass over state,
    /// keeping values in zmm registers between add-constant and x^7.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline(always)]
    fn full_rounds(state: &mut [Self; 12], round_ctr: &mut usize) {
        use crate::hash::poseidon::{ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS};
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
        use plonky2_field::packed::PackedField;
        use plonky2_field::ops::Square;

        for _ in 0..HALF_N_FULL_ROUNDS {
            let base = *round_ctr * 12;

            // --- Interleaved constant + sbox for all 12 elements ---
            // Load scalar constants and start scalar add for 8..12 (independent of packed load)
            let c8 = GoldilocksField(ALL_ROUND_CONSTANTS[base + 8]);
            let c9 = GoldilocksField(ALL_ROUND_CONSTANTS[base + 9]);
            let c10 = GoldilocksField(ALL_ROUND_CONSTANTS[base + 10]);
            let c11 = GoldilocksField(ALL_ROUND_CONSTANTS[base + 11]);
            let sx8 = state[8] + c8;
            let sx9 = state[9] + c9;
            let sx10 = state[10] + c10;
            let sx11 = state[11] + c11;

            // Load packed constants and start packed add for 0..8
            let mut packed = *P::from_slice(&state[0..8]);
            let mut constants = P::ZEROS;
            for i in 0..8 {
                constants.0[i] = GoldilocksField(ALL_ROUND_CONSTANTS[base + i]);
            }
            packed = packed + constants;

            // Interleave packed sbox with scalar sbox to use both FMA and integer mul units
            let px2 = packed.square();
            let sx8_2 = sx8.square();
            let sx9_2 = sx9.square();

            let px4 = px2.square();
            let sx10_2 = sx10.square();
            let sx11_2 = sx11.square();

            let px3 = packed * px2;
            let sx8_4 = sx8_2.square();
            let sx9_4 = sx9_2.square();

            let px7 = px3 * px4;
            let sx10_4 = sx10_2.square();
            let sx11_4 = sx11_2.square();

            // Store packed results
            state[0] = px7.0[0];
            state[1] = px7.0[1];
            state[2] = px7.0[2];
            state[3] = px7.0[3];
            state[4] = px7.0[4];
            state[5] = px7.0[5];
            state[6] = px7.0[6];
            state[7] = px7.0[7];

            // Finish scalar sbox
            let sx8_3 = sx8 * sx8_2;
            let sx9_3 = sx9 * sx9_2;
            let sx10_3 = sx10 * sx10_2;
            let sx11_3 = sx11 * sx11_2;
            state[8] = sx8_3 * sx8_4;
            state[9] = sx9_3 * sx9_4;
            state[10] = sx10_3 * sx10_4;
            state[11] = sx11_3 * sx11_4;

            // MDS layer (unchanged)
            *state = Self::mds_layer(state);
            *round_ctr += 1;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline(always)]
    #[unroll::unroll_for_loops]
    fn sbox_layer(state: &mut [Self; 12]) {
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
        use plonky2_field::packed::PackedField;
        use plonky2_field::ops::Square;

        // Process first 8 elements using AVX-512 packed field (8-wide SIMD)
        let packed = *P::from_slice(&state[0..8]);
        let x2 = packed.square();
        let x4 = x2.square();
        let x3 = packed * x2;
        let x7 = x3 * x4;
        state[0] = x7.0[0];
        state[1] = x7.0[1];
        state[2] = x7.0[2];
        state[3] = x7.0[3];
        state[4] = x7.0[4];
        state[5] = x7.0[5];
        state[6] = x7.0[6];
        state[7] = x7.0[7];

        // Process last 4 elements scalar (padding overhead exceeds SIMD benefit)
        for i in 8..12 {
            let x = state[i];
            let x2 = x.square();
            let x4 = x2.square();
            let x3 = x * x2;
            state[i] = x3 * x4;
        }
    }

    /// AVX-512 optimized sparse MDS for partial rounds.
    /// The multiply-accumulate step (result[i] = state[i] + state[0] * v[i])
    /// processes 8 elements at a time using packed field multiplication.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[inline(always)]
    fn mds_partial_layer_fast(state: &[Self; 12], r: usize) -> [Self; 12] {
        use crate::field::types::{Field, PrimeField64};

        // Step 1: Compute d using scalar u160 accumulation (unchanged)
        let mut d_sum = (0u128, 0u32);
        for i in 1..12 {
            let t = Self::FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
            let si = state[i].to_noncanonical_u64() as u128;
            d_sum = crate::hash::poseidon::add_u160_u128(d_sum, si * t);
        }
        let s0 = state[0].to_noncanonical_u64() as u128;
        let mds0to0 = (Self::MDS_MATRIX_CIRC[0] + Self::MDS_MATRIX_DIAG[0]) as u128;
        d_sum = crate::hash::poseidon::add_u160_u128(d_sum, s0 * mds0to0);
        let d = crate::hash::poseidon::reduce_u160::<Self>(d_sum);

        // Step 2: AVX-512 packed multiply-accumulate for result[1..12]
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
        use plonky2_field::packed::PackedField;

        let mut result = [Self::ZERO; 12];
        result[0] = d;

        let v = Self::FAST_PARTIAL_ROUND_VS[r];

        // Broadcast state[0] to all 8 lanes
        let s0_broadcast = P([state[0]; 8]);

        // Pack v[1..9] into a zmm
        let mut v_packed = P::ZEROS;
        for i in 0..8 {
            v_packed.0[i] = GoldilocksField::from_canonical_u64(v[i]);
        }

        // Pack state[1..9] into a zmm
        let mut st_packed = P::ZEROS;
        for i in 0..8 {
            st_packed.0[i] = state[i + 1];
        }

        // Compute s0 * v[1..9] + state[1..9] using packed field ops
        let prod = s0_broadcast * v_packed;
        let result_packed = prod + st_packed;
        for i in 0..8 {
            result[i + 1] = result_packed.0[i];
        }

        // Last 3 elements: scalar
        for i in 9..12 {
            let t = Self::from_canonical_u64(v[i - 1]);
            result[i] = state[i].multiply_accumulate(state[0], t);
        }

        result
    }

    #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    #[inline(always)]
    fn sbox_layer(state: &mut [Self; 12]) {
        unsafe {
            crate::hash::arch::aarch64::poseidon_goldilocks_neon::sbox_layer(state);
        }
    }

    #[cfg(all(target_arch="aarch64", target_feature="neon"))]
    #[inline(always)]
    fn mds_layer(state: &[Self; 12]) -> [Self; 12] {
        unsafe {
            crate::hash::arch::aarch64::poseidon_goldilocks_neon::mds_layer(state)
        }
    }
}

// MDS layer helper methods
// The following code has been adapted from winterfell/crypto/src/hash/mds/mds_f64_12x12.rs
// located at https://github.com/facebook/winterfell.
#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
mod poseidon12_mds {
    pub(crate) const MDS_FREQ_BLOCK_ONE: [i64; 3] = [16, 32, 16];
    pub(crate) const MDS_FREQ_BLOCK_TWO: [(i64, i64); 3] = [(2, -1), (-4, 1), (16, 1)];
    pub(crate) const MDS_FREQ_BLOCK_THREE: [i64; 3] = [-1, -8, 2];

    /// Split 3 x 4 FFT-based MDS vector-multiplication with the Poseidon circulant MDS matrix.
    #[inline(always)]
    pub(crate) const fn mds_multiply_freq(state: [u64; 12]) -> [u64; 12] {
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = state;

        let (u0, u1, u2) = fft4_real([s0, s3, s6, s9]);
        let (u4, u5, u6) = fft4_real([s1, s4, s7, s10]);
        let (u8, u9, u10) = fft4_real([s2, s5, s8, s11]);

        // This where the multiplication in frequency domain is done. More precisely, and with
        // the appropriate permutations in between, the sequence of
        // 3-point FFTs --> multiplication by twiddle factors --> Hadamard multiplication -->
        // 3 point iFFTs --> multiplication by (inverse) twiddle factors
        // is "squashed" into one step composed of the functions "block1", "block2" and "block3".
        // The expressions in the aforementioned functions are the result of explicit computations
        // combined with the Karatsuba trick for the multiplication of complex numbers.

        let [v0, v4, v8] = block1([u0, u4, u8], MDS_FREQ_BLOCK_ONE);
        let [v1, v5, v9] = block2([u1, u5, u9], MDS_FREQ_BLOCK_TWO);
        let [v2, v6, v10] = block3([u2, u6, u10], MDS_FREQ_BLOCK_THREE);
        // The 4th block is not computed as it is similar to the 2nd one, up to complex conjugation.

        let [s0, s3, s6, s9] = ifft4_real_unreduced((v0, v1, v2));
        let [s1, s4, s7, s10] = ifft4_real_unreduced((v4, v5, v6));
        let [s2, s5, s8, s11] = ifft4_real_unreduced((v8, v9, v10));

        [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    }

    #[inline(always)]
    pub(crate) const fn block1(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 + x1 * y2 + x2 * y1;
        let z1 = x0 * y1 + x1 * y0 + x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;

        [z0, z1, z2]
    }

    #[inline(always)]
    pub(crate) const fn block2(x: [(i64, i64); 3], y: [(i64, i64); 3]) -> [(i64, i64); 3] {
        let [(x0r, x0i), (x1r, x1i), (x2r, x2i)] = x;
        let [(y0r, y0i), (y1r, y1i), (y2r, y2i)] = y;
        let x0s = x0r + x0i;
        let x1s = x1r + x1i;
        let x2s = x2r + x2i;
        let y0s = y0r + y0i;
        let y1s = y1r + y1i;
        let y2s = y2r + y2i;

        // Compute x0​y0 ​− ix1​y2​ − ix2​y1​ using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y0r, x0i * y0i);
        let m1 = (x1r * y2r, x1i * y2i);
        let m2 = (x2r * y1r, x2i * y1i);
        let z0r = (m0.0 - m0.1) + (x1s * y2s - m1.0 - m1.1) + (x2s * y1s - m2.0 - m2.1);
        let z0i = (x0s * y0s - m0.0 - m0.1) + (-m1.0 + m1.1) + (-m2.0 + m2.1);
        let z0 = (z0r, z0i);

        // Compute x0​y1​ + x1​y0​ − ix2​y2 using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y1r, x0i * y1i);
        let m1 = (x1r * y0r, x1i * y0i);
        let m2 = (x2r * y2r, x2i * y2i);
        let z1r = (m0.0 - m0.1) + (m1.0 - m1.1) + (x2s * y2s - m2.0 - m2.1);
        let z1i = (x0s * y1s - m0.0 - m0.1) + (x1s * y0s - m1.0 - m1.1) + (-m2.0 + m2.1);
        let z1 = (z1r, z1i);

        // Compute x0​y2​ + x1​y1 ​+ x2​y0​ using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y2r, x0i * y2i);
        let m1 = (x1r * y1r, x1i * y1i);
        let m2 = (x2r * y0r, x2i * y0i);
        let z2r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1);
        let z2i = (x0s * y2s - m0.0 - m0.1) + (x1s * y1s - m1.0 - m1.1) + (x2s * y0s - m2.0 - m2.1);
        let z2 = (z2r, z2i);

        [z0, z1, z2]
    }

    #[inline(always)]
    pub(crate) const fn block3(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 - x1 * y2 - x2 * y1;
        let z1 = x0 * y1 + x1 * y0 - x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;

        [z0, z1, z2]
    }

    /// Real 2-FFT over u64 integers.
    #[inline(always)]
    pub(crate) const fn fft2_real(x: [u64; 2]) -> [i64; 2] {
        [(x[0] as i64 + x[1] as i64), (x[0] as i64 - x[1] as i64)]
    }

    /// Real 2-iFFT over u64 integers.
    /// Division by two to complete the inverse FFT is not performed here.
    #[inline(always)]
    pub(crate) const fn ifft2_real_unreduced(y: [i64; 2]) -> [u64; 2] {
        [(y[0] + y[1]) as u64, (y[0] - y[1]) as u64]
    }

    /// Real 4-FFT over u64 integers.
    #[inline(always)]
    pub(crate) const fn fft4_real(x: [u64; 4]) -> (i64, (i64, i64), i64) {
        let [z0, z2] = fft2_real([x[0], x[2]]);
        let [z1, z3] = fft2_real([x[1], x[3]]);
        let y0 = z0 + z1;
        let y1 = (z2, -z3);
        let y2 = z0 - z1;
        (y0, y1, y2)
    }

    /// Real 4-iFFT over u64 integers.
    /// Division by four to complete the inverse FFT is not performed here.
    #[inline(always)]
    pub(crate) const fn ifft4_real_unreduced(y: (i64, (i64, i64), i64)) -> [u64; 4] {
        let z0 = y.0 + y.2;
        let z1 = y.0 - y.2;
        let z2 = y.1 .0;
        let z3 = -y.1 .1;

        let [x0, x2] = ifft2_real_unreduced([z0, z2]);
        let [x1, x3] = ifft2_real_unreduced([z1, z3]);

        [x0, x1, x2, x3]
    }
}

/// Batched Poseidon permutation processing 8 states simultaneously using AVX-512 SoA layout.
///
/// Layout: 8 independent [u64; 12] states are stored as 12 __m512i registers,
/// where zmm[j] holds element j from all 8 states.
///
/// All operations (constant add, sbox, MDS) are element-wise and parallelize
/// across the 8 lanes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
pub(crate) mod batched_poseidon {
    use super::{GoldilocksField, poseidon12_mds, N_PARTIAL_ROUNDS};
    use crate::hash::poseidon::{ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, SPONGE_WIDTH, Poseidon};
    use crate::field::types::Field;
    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::ops::Square;
    use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
    use plonky2_field::packed::PackedField;
    use core::arch::x86_64::{__m512i, _mm512_add_epi64, _mm512_set1_epi64, _mm512_sub_epi64,
                                _mm512_mullo_epi64, _mm512_and_si512, _mm512_srli_epi64,
                                _mm512_slli_epi64, _mm512_loadu_si512, _mm512_storeu_si512};

    /// Broadcast a single u64 constant to all 8 lanes of a zmm register.
    #[inline(always)]
    fn broadcast_i64(val: i64) -> __m512i {
        unsafe { _mm512_set1_epi64(val) }
    }

    /// Broadcast a single u64 constant as u64 to all 8 lanes.
    #[inline(always)]
    fn broadcast_u64(val: u64) -> __m512i {
        // Safety: bit pattern is the same for u64 and i64 in two's complement
        broadcast_i64(val as i64)
    }

    /// Packed FFT4 on 8 independent 4-element u64 arrays.
    /// Input: 4 __m512i registers, each holding 8 u64 values.
    /// Output: (y0, (y1r, y1i), y2) as packed i64 values.
    #[inline(always)]
    fn fft4_real_packed(x: [__m512i; 4]) -> (__m512i, (__m512i, __m512i), __m512i) {
        unsafe {
            // fft2_real([x[0], x[2]]): cast u64 to i64, then z0=x[0]+x[2], z2=x[0]-x[2]
            let z0_02 = _mm512_add_epi64(x[0], x[2]);
            let z2_02 = _mm512_sub_epi64(x[0], x[2]);
            // fft2_real([x[1], x[3]]): z1=x[1]+x[3], z3=x[1]-x[3]
            let z1_13 = _mm512_add_epi64(x[1], x[3]);
            let z3_13 = _mm512_sub_epi64(x[1], x[3]);
            // y0 = z0_02 + z1_13
            // y1 = (z2_02, -z3_13)
            // y2 = z0_02 - z1_13
            let y0 = _mm512_add_epi64(z0_02, z1_13);
            let y1r = z2_02;
            let y1i = _mm512_sub_epi64(_mm512_set1_epi64(0), z3_13); // -z3_13
            let y2 = _mm512_sub_epi64(z0_02, z1_13);
            (y0, (y1r, y1i), y2)
        }
    }

    /// Packed IFFT4 on 8 independent triples.
    /// Division by four is NOT performed (unreduced).
    #[inline(always)]
    fn ifft4_real_unreduced_packed(y: (__m512i, (__m512i, __m512i), __m512i)) -> [__m512i; 4] {
        unsafe {
            let z0 = _mm512_add_epi64(y.0, y.2);
            let z1 = _mm512_sub_epi64(y.0, y.2);
            let z2 = (y.1).0;
            let z3 = _mm512_sub_epi64(_mm512_set1_epi64(0), (y.1).1); // -y.1.1
            // ifft2_real_unreduced([z0, z2]): x0 = z0 + z2, x2 = z0 - z2
            let x0 = _mm512_add_epi64(z0, z2);
            let x2 = _mm512_sub_epi64(z0, z2);
            // ifft2_real_unreduced([z1, z3]): x1 = z1 + z3, x3 = z1 - z3
            let x1 = _mm512_add_epi64(z1, z3);
            let x3 = _mm512_sub_epi64(z1, z3);
            [x0, x1, x2, x3]
        }
    }

    /// Packed block1: circulant multiply in frequency domain.
    /// z0 = x0*y0 + x1*y2 + x2*y1
    /// z1 = x0*y1 + x1*y0 + x2*y2
    /// z2 = x0*y2 + x1*y1 + x2*y0
    /// Using mullo_epi64 for low-64-bit multiply.
    #[inline(always)]
    fn block1_packed(x: [__m512i; 3], y: [i64; 3]) -> [__m512i; 3] {
        unsafe {
            let y0 = broadcast_i64(y[0]);
            let y1 = broadcast_i64(y[1]);
            let y2 = broadcast_i64(y[2]);
            // Use mullo for low-64-bit product (same as wrapping i64 mul)
            let x0_y0 = _mm512_mullo_epi64(x[0], y0);
            let x1_y2 = _mm512_mullo_epi64(x[1], y2);
            let x2_y1 = _mm512_mullo_epi64(x[2], y1);
            let z0 = _mm512_add_epi64(_mm512_add_epi64(x0_y0, x1_y2), x2_y1);

            let x0_y1 = _mm512_mullo_epi64(x[0], y1);
            let x1_y0 = _mm512_mullo_epi64(x[1], y0);
            let x2_y2 = _mm512_mullo_epi64(x[2], y2);
            let z1 = _mm512_add_epi64(_mm512_add_epi64(x0_y1, x1_y0), x2_y2);

            let x0_y2 = _mm512_mullo_epi64(x[0], y2);
            let x1_y1 = _mm512_mullo_epi64(x[1], y1);
            let x2_y0 = _mm512_mullo_epi64(x[2], y0);
            let z2 = _mm512_add_epi64(_mm512_add_epi64(x0_y2, x1_y1), x2_y0);

            [z0, z1, z2]
        }
    }

    /// Packed block3: circulant multiply with subtraction.
    /// z0 = x0*y0 - x1*y2 - x2*y1
    /// z1 = x0*y1 + x1*y0 - x2*y2
    /// z2 = x0*y2 + x1*y1 + x2*y0
    #[inline(always)]
    fn block3_packed(x: [__m512i; 3], y: [i64; 3]) -> [__m512i; 3] {
        unsafe {
            let y0 = broadcast_i64(y[0]);
            let y1 = broadcast_i64(y[1]);
            let y2 = broadcast_i64(y[2]);

            let x0_y0 = _mm512_mullo_epi64(x[0], y0);
            let x1_y2 = _mm512_mullo_epi64(x[1], y2);
            let x2_y1 = _mm512_mullo_epi64(x[2], y1);
            let z0 = _mm512_sub_epi64(_mm512_sub_epi64(x0_y0, x1_y2), x2_y1);

            let x0_y1 = _mm512_mullo_epi64(x[0], y1);
            let x1_y0 = _mm512_mullo_epi64(x[1], y0);
            let x2_y2 = _mm512_mullo_epi64(x[2], y2);
            let z1 = _mm512_add_epi64(_mm512_add_epi64(x0_y1, x1_y0), _mm512_sub_epi64(_mm512_set1_epi64(0), x2_y2));

            let x0_y2 = _mm512_mullo_epi64(x[0], y2);
            let x1_y1 = _mm512_mullo_epi64(x[1], y1);
            let x2_y0 = _mm512_mullo_epi64(x[2], y0);
            let z2 = _mm512_add_epi64(_mm512_add_epi64(x0_y2, x1_y1), x2_y0);

            [z0, z1, z2]
        }
    }

    /// Packed block2: complex-valued circulant multiply with Karatsuba.
    /// Same logic as scalar block2 but on packed i64.
    #[inline(always)]
    fn block2_packed(
        x: [(__m512i, __m512i); 3],
        y: [(i64, i64); 3],
    ) -> [(__m512i, __m512i); 3] {
        unsafe {
            let [(x0r, x0i), (x1r, x1i), (x2r, x2i)] = x;
            let [(y0r, y0i), (y1r, y1i), (y2r, y2i)] = y;

            let x0s = _mm512_add_epi64(x0r, x0i);
            let x1s = _mm512_add_epi64(x1r, x1i);
            let x2s = _mm512_add_epi64(x2r, x2i);
            let y0s = broadcast_i64(y0r + y0i);
            let y1s = broadcast_i64(y1r + y1i);
            let y2s = broadcast_i64(y2r + y2i);

            let by0r = broadcast_i64(y0r);
            let by0i = broadcast_i64(y0i);
            let by1r = broadcast_i64(y1r);
            let by1i = broadcast_i64(y1i);
            let by2r = broadcast_i64(y2r);
            let by2i = broadcast_i64(y2i);

            // z0 computation
            let m0r = _mm512_mullo_epi64(x0r, by0r);
            let m0i = _mm512_mullo_epi64(x0i, by0i);
            let m1r = _mm512_mullo_epi64(x1r, by2r);
            let m1i = _mm512_mullo_epi64(x1i, by2i);
            let m2r = _mm512_mullo_epi64(x2r, by1r);
            let m2i = _mm512_mullo_epi64(x2i, by1i);
            let x1s_y2s = _mm512_mullo_epi64(x1s, y2s);
            let x2s_y1s = _mm512_mullo_epi64(x2s, y1s);
            let x0s_y0s = _mm512_mullo_epi64(x0s, y0s);
            let z0r = _mm512_add_epi64(
                _mm512_sub_epi64(m0r, m0i),
                _mm512_add_epi64(
                    _mm512_sub_epi64(x1s_y2s, _mm512_add_epi64(m1r, m1i)),
                    _mm512_sub_epi64(x2s_y1s, _mm512_add_epi64(m2r, m2i)),
                ),
            );
            let z0i = _mm512_add_epi64(
                _mm512_sub_epi64(x0s_y0s, _mm512_add_epi64(m0r, m0i)),
                _mm512_add_epi64(
                    _mm512_sub_epi64(m1i, m1r),
                    _mm512_sub_epi64(m2i, m2r),
                ),
            );

            // z1 computation
            let m0r = _mm512_mullo_epi64(x0r, by1r);
            let m0i = _mm512_mullo_epi64(x0i, by1i);
            let m1r = _mm512_mullo_epi64(x1r, by0r);
            let m1i = _mm512_mullo_epi64(x1i, by0i);
            let m2r = _mm512_mullo_epi64(x2r, by2r);
            let m2i = _mm512_mullo_epi64(x2i, by2i);
            let x2s_y2s = _mm512_mullo_epi64(x2s, y2s);
            let x1s_y0s = _mm512_mullo_epi64(x1s, y0s);
            let x0s_y1s = _mm512_mullo_epi64(x0s, y1s);
            let z1r = _mm512_add_epi64(
                _mm512_add_epi64(
                    _mm512_sub_epi64(m0r, m0i),
                    _mm512_sub_epi64(m1r, m1i),
                ),
                _mm512_sub_epi64(x2s_y2s, _mm512_add_epi64(m2r, m2i)),
            );
            let z1i = _mm512_add_epi64(
                _mm512_sub_epi64(x0s_y1s, _mm512_add_epi64(m0r, m0i)),
                _mm512_add_epi64(
                    _mm512_sub_epi64(x1s_y0s, _mm512_add_epi64(m1r, m1i)),
                    _mm512_sub_epi64(m2i, m2r),
                ),
            );

            // z2 computation
            let m0r = _mm512_mullo_epi64(x0r, by2r);
            let m0i = _mm512_mullo_epi64(x0i, by2i);
            let m1r = _mm512_mullo_epi64(x1r, by1r);
            let m1i = _mm512_mullo_epi64(x1i, by1i);
            let m2r = _mm512_mullo_epi64(x2r, by0r);
            let m2i = _mm512_mullo_epi64(x2i, by0i);
            let x0s_y2s = _mm512_mullo_epi64(x0s, y2s);
            let x1s_y1s = _mm512_mullo_epi64(x1s, y1s);
            let x2s_y0s = _mm512_mullo_epi64(x2s, y0s);
            let z2r = _mm512_add_epi64(
                _mm512_add_epi64(
                    _mm512_sub_epi64(m0r, m0i),
                    _mm512_sub_epi64(m1r, m1i),
                ),
                _mm512_sub_epi64(m2r, m2i),
            );
            let z2i = _mm512_add_epi64(
                _mm512_add_epi64(
                    _mm512_sub_epi64(x0s_y2s, _mm512_add_epi64(m0r, m0i)),
                    _mm512_sub_epi64(x1s_y1s, _mm512_add_epi64(m1r, m1i)),
                ),
                _mm512_sub_epi64(x2s_y0s, _mm512_add_epi64(m2r, m2i)),
            );

            [(z0r, z0i), (z1r, z1i), (z2r, z2i)]
        }
    }

    /// Packed MDS frequency-domain circulant multiply on 8 states simultaneously.
    /// Input/Output: 12 __m512i registers in SoA layout (zmm[j] = element j from 8 states).
    /// Operates on u64 values split into low/high 32-bit halves.
    #[inline(always)]
    #[allow(dead_code)]
    fn mds_multiply_freq_packed(state: &[__m512i; 12]) -> [__m512i; 12] {
        // Split each u64 into low 32 bits and high 32 bits
        let mask32 = unsafe { _mm512_set1_epi64(0xFFFFFFFF) };
        let mut lo = [unsafe { _mm512_set1_epi64(0) }; 12];
        let mut hi = [unsafe { _mm512_set1_epi64(0) }; 12];
        for i in 0..12 {
            // lo[i] = state[i] & 0xFFFFFFFF
            lo[i] = unsafe { _mm512_and_si512(state[i], mask32) };
            // hi[i] = state[i] >> 32 (arithmetic shift)
            hi[i] = unsafe { _mm512_srli_epi64::<32>(state[i]) };
        }

        // Apply MDS multiply to both halves
        let lo = mds_multiply_freq_raw(&lo);
        let hi = mds_multiply_freq_raw(&hi);

        // Combine: result = lo + (hi << 32)
        let mut result = [unsafe { _mm512_set1_epi64(0) }; 12];
        for i in 0..12 {
            // Shift hi left by 32 bits
            let hi_shifted = unsafe { _mm512_slli_epi64(hi[i], 32) };
            result[i] = unsafe { _mm512_add_epi64(lo[i], hi_shifted) };
        }
        result
    }

    /// Raw packed MDS frequency multiply on 12 i64 values.
    /// This is the same as poseidon12_mds::mds_multiply_freq but on packed data.
    #[inline(always)]
    fn mds_multiply_freq_raw(state: &[__m512i; 12]) -> [__m512i; 12] {
        use poseidon12_mds::{MDS_FREQ_BLOCK_ONE, MDS_FREQ_BLOCK_TWO, MDS_FREQ_BLOCK_THREE};

        // FFT4 on groups [0,3,6,9], [1,4,7,10], [2,5,8,11]
        // fft4_real returns (y0, (y1r, y1i), y2)
        let (u0, (u1r, u1i), u2) = fft4_real_packed([state[0], state[3], state[6], state[9]]);
        let (u4, (u5r, u5i), u6) = fft4_real_packed([state[1], state[4], state[7], state[10]]);
        let (u8, (u9r, u9i), u10) = fft4_real_packed([state[2], state[5], state[8], state[11]]);

        // Block multiplies
        // block1: real-valued circulant on (u0, u4, u8)
        let [v0, v4, v8] = block1_packed([u0, u4, u8], MDS_FREQ_BLOCK_ONE);
        // block2: complex-valued circulant on ((u1r,u1i), (u5r,u5i), (u9r,u9i))
        let [(v1r, v1i), (v5r, v5i), (v9r, v9i)] = block2_packed(
            [(u1r, u1i), (u5r, u5i), (u9r, u9i)],
            MDS_FREQ_BLOCK_TWO,
        );
        // block3: real-valued circulant on (u2, u6, u10)
        let [v2, v6, v10] = block3_packed([u2, u6, u10], MDS_FREQ_BLOCK_THREE);

        // IFFT4
        let [s0, s3, s6, s9] = ifft4_real_unreduced_packed((v0, (v1r, v1i), v2));
        let [s1, s4, s7, s10] = ifft4_real_unreduced_packed((v4, (v5r, v5i), v6));
        let [s2, s5, s8, s11] = ifft4_real_unreduced_packed((v8, (v9r, v9i), v10));

        [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    }

    /// Complete batched MDS layer: circulant multiply + diagonal + field reduction.
    /// Works on 8 states in SoA layout using Avx512GoldilocksField (packed field ops).
    #[inline(always)]
    pub fn mds_layer_batch(packed: &[P; 12]) -> [P; 12] {
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::{reduce128, mul64_64, EPSILON};
        use crate::field::goldilocks_field::GoldilocksField;

        // Extract raw u64 values from packed field elements
        let raw: [__m512i; 12] = core::array::from_fn(|j| packed[j].get());

        // Store original state[0] for diagonal addition
        let orig_s0 = raw[0];

        // Split each u64 into low 32 bits and high 32 bits
        let mask32 = unsafe { _mm512_set1_epi64(0xFFFFFFFF) };
        let mut lo = [unsafe { _mm512_set1_epi64(0) }; 12];
        let mut hi = [unsafe { _mm512_set1_epi64(0) }; 12];
        for i in 0..12 {
            lo[i] = unsafe { _mm512_and_si512(raw[i], mask32) };
            hi[i] = unsafe { _mm512_srli_epi64::<32>(raw[i]) };
        }

        // Apply circulant MDS to both halves
        let lo = mds_multiply_freq_raw(&lo);
        let hi = mds_multiply_freq_raw(&hi);

        // Combine and reduce using proper 128-bit arithmetic.
        // Scalar: s = lo + (hi << 32) as u128; from_noncanonical_u96((s as u64, (s >> 64) as u32))
        // Packed: split hi into lo_32 and hi_above_32, construct 128-bit (hi_above_32, lo + hi_lo_32 << 32)
        let mask32 = unsafe { _mm512_set1_epi64(0xFFFFFFFF) };
        let mut result = [P::ZEROS; 12];
        for i in 0..12 {
            // hi_lo32 = hi[i] & 0xFFFFFFFF (lower 32 bits of hi_result)
            let hi_lo32 = unsafe { _mm512_and_si512(hi[i], mask32) };
            // hi_above32 = hi[i] >> 32 (upper bits of hi_result, fits in 32 bits for MDS)
            let hi_above32 = unsafe { _mm512_srli_epi64::<32>(hi[i]) };
            // combined_lo = lo[i] + (hi_lo32 << 32) — both fit in 64 bits
            let hi_lo32_shifted = unsafe { _mm512_slli_epi64::<32>(hi_lo32) };
            let combined_lo = unsafe { _mm512_add_epi64(lo[i], hi_lo32_shifted) };
            // combined_hi = hi_above32 (no carry possible since MDS outputs are bounded)
            // The scalar code does from_noncanonical_u96((s as u64, (s >> 64) as u32))
            // which is reduce96((combined_lo, combined_hi as u32))
            // We can use reduce128 which does the same thing for 128-bit values
            let reduced = unsafe { reduce128((hi_above32, combined_lo)) };
            result[i] = P::new(reduced);
        }

        use crate::hash::poseidon::Poseidon;
        // Add diagonal: result[0] += from_noncanonical_u96(MDS_MATRIX_DIAG[0] * state[0])
        let diag_val = <super::GoldilocksField as Poseidon>::MDS_MATRIX_DIAG[0];
        let diag = unsafe { broadcast_u64(diag_val) };
        let diag_times_s0 = unsafe { reduce128(mul64_64(diag, orig_s0)) };
        result[0] = result[0] + P::new(diag_times_s0);

        result
    }

    /// Batched full rounds: constant_layer + sbox_layer + mds_layer for 8 states.
    /// Processes `n_rounds` full Poseidon rounds on packed SoA data.
    #[inline(always)]
    fn full_rounds_batch(packed: &mut [P; 12], round_ctr: &mut usize, n_rounds: usize) {
        for _ in 0..n_rounds {
            let base = *round_ctr * 12;

            // Fused constant + sbox for all 12 elements
            // Interleave: do constant add and sbox together to keep data in zmm registers
            for j in 0..12 {
                let c = GoldilocksField(ALL_ROUND_CONSTANTS[base + j]);
                packed[j] = packed[j] + P([c, c, c, c, c, c, c, c]);
                // Sbox: x^7
                let x2 = packed[j].square();
                let x4 = x2.square();
                let x3 = packed[j] * x2;
                packed[j] = x3 * x4;
            }

            // MDS layer: packed circulant multiply + diagonal + reduce
            let new_state = mds_layer_batch(packed);
            *packed = new_state;

            *round_ctr += 1;
        }
    }

    /// Batched partial rounds for 8 states simultaneously.
    /// Processes: partial_first_constant_layer + mds_partial_layer_init + 22 sparse rounds.
    #[inline(always)]
    fn partial_rounds_batch(packed: &mut [P; 12], round_ctr: &mut usize) {
        // 1. partial_first_constant_layer: add FAST_PARTIAL_FIRST_ROUND_CONSTANT to all elements
        for j in 0..12 {
            let c = GoldilocksField::from_canonical_u64(
                <GoldilocksField as Poseidon>::FAST_PARTIAL_FIRST_ROUND_CONSTANT[j]
            );
            packed[j] = packed[j] + P([c, c, c, c, c, c, c, c]);
        }

        // 2. mds_partial_layer_init: 11x11 matrix-vector multiply
        // result[0] = state[0], result[c] = Σ_{r=1}^{11} state[r] * matrix[r-1][c-1]
        // Process using packed field operations on all 8 states simultaneously
        let mut result: [P; 12] = [P::ZEROS; 12];
        result[0] = packed[0]; // result[0] = state[0] (no change)

        // For columns 1..8: use packed multiply-accumulate
        // Pre-compute packed matrix values
        // Matrix is [11][8] for columns 1..8 of rows 1..11
        let mut packed_matrix: [[P; 8]; 11] = [[P::ZEROS; 8]; 11];
        for r in 1..12 {
            for c in 0..8 {
                let mat_val = GoldilocksField::from_canonical_u64(
                    <GoldilocksField as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c]
                );
                packed_matrix[r - 1][c] = P([mat_val, mat_val, mat_val, mat_val, mat_val, mat_val, mat_val, mat_val]);
            }
        }

        let mut acc = [P::ZEROS; 8]; // 8 accumulators for columns 1..8
        for r in 1..12 {
            for c in 0..8 {
                acc[c] = acc[c] + packed[r] * packed_matrix[r - 1][c];
            }
        }
        // Store packed results for columns 1..8
        for c in 0..8 {
            result[c + 1] = acc[c];
        }

        // For columns 9..11: scalar matrix multiply (3 extra columns per state)
        for c in 8..11 {
            for r in 1..12 {
                let mat_val = GoldilocksField::from_canonical_u64(
                    <GoldilocksField as Poseidon>::FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c]
                );
                result[c + 1] = result[c + 1] + packed[r] * P([mat_val, mat_val, mat_val, mat_val, mat_val, mat_val, mat_val, mat_val]);
            }
        }
        *packed = result;

        // 3. 22 sparse partial rounds
        for i in 0..N_PARTIAL_ROUNDS {
            // Sbox on element 0: x^7
            let x2 = packed[0].square();
            let x4 = x2.square();
            let x3 = packed[0] * x2;
            packed[0] = x3 * x4;

            // Add round constant to element 0 (using add_canonical_u64 semantics)
            // FAST_PARTIAL_ROUND_CONSTANTS are already canonical Goldilocks values.
            // The scalar code uses state[0].add_canonical_u64(c) which is just += c
            // when c < field modulus. Since these constants are pre-reduced, += is fine.
            let c = GoldilocksField(<GoldilocksField as Poseidon>::FAST_PARTIAL_ROUND_CONSTANTS[i]);
            packed[0] = packed[0] + P([c, c, c, c, c, c, c, c]);

            // Sparse MDS (mds_partial_layer_fast):
            // d = state[0] * mds0to0 + Σ_{r=1}^{11} state[r] * w_hats[r-1]
            // result[0] = d
            // result[r] = state[r] + state[0] * v[r-1] for r=1..11
            let mds0to0 = GoldilocksField::from_canonical_u64(
                <GoldilocksField as Poseidon>::MDS_MATRIX_CIRC[0]
                + <GoldilocksField as Poseidon>::MDS_MATRIX_DIAG[0]
            );

            // Compute d using packed multiply-accumulate
            let mut d = packed[0] * P([mds0to0, mds0to0, mds0to0, mds0to0, mds0to0, mds0to0, mds0to0, mds0to0]);
            for r in 1..12 {
                let w = GoldilocksField::from_canonical_u64(
                    <GoldilocksField as Poseidon>::FAST_PARTIAL_ROUND_W_HATS[i][r - 1]
                );
                d = d + packed[r] * P([w, w, w, w, w, w, w, w]);
            }

            // Apply sparse MDS
            let old_s0 = packed[0];
            packed[0] = d;
            let v = <GoldilocksField as Poseidon>::FAST_PARTIAL_ROUND_VS[i];
            for r in 1..12 {
                let v_val = GoldilocksField::from_canonical_u64(v[r - 1]);
                packed[r] = packed[r] + old_s0 * P([v_val, v_val, v_val, v_val, v_val, v_val, v_val, v_val]);
            }
        }

        *round_ctr += N_PARTIAL_ROUNDS;
    }

    /// Batched permutation for 8 states simultaneously.
    /// Takes 8 × [u64; 12] raw state values and returns 8 permuted states.
    pub fn permute_batch_8(states: &[[u64; 12]; 8]) -> [[u64; 12]; 8] {
        // Convert AoS → SoA: 12 × P (each P holds 8 values at same position)
        let mut packed: [P; 12] = core::array::from_fn(|j| {
            let vals: [GoldilocksField; 8] = core::array::from_fn(|i| GoldilocksField(states[i][j]));
            P(vals)
        });

        let mut round_ctr = 0;

        // First 4 full rounds
        full_rounds_batch(&mut packed, &mut round_ctr, HALF_N_FULL_ROUNDS);

        // Batched partial rounds
        partial_rounds_batch(&mut packed, &mut round_ctr);

        // Last 4 full rounds
        full_rounds_batch(&mut packed, &mut round_ctr, HALF_N_FULL_ROUNDS);

        // Convert SoA → AoS
        let mut result = [[0u64; 12]; 8];
        for j in 0..12 {
            for i in 0..8 {
                result[i][j] = packed[j].0[i].0;
            }
        }
        result
    }

    /// Packed permutation: takes state already in SoA form (packed), returns packed result.
    /// Avoids AoS→SoA→AoS conversion overhead.
    #[inline]
    pub fn permute_packed_8(packed: &[P; 12]) -> [P; 12] {
        let mut packed = *packed;
        let mut round_ctr = 0;

        // First 4 full rounds
        full_rounds_batch(&mut packed, &mut round_ctr, HALF_N_FULL_ROUNDS);

        // Batched partial rounds
        partial_rounds_batch(&mut packed, &mut round_ctr);

        // Last 4 full rounds
        full_rounds_batch(&mut packed, &mut round_ctr, HALF_N_FULL_ROUNDS);

        packed
    }

    /// Test entry: batch MDS (circulant part only) for 8 states.
    /// Returns the raw circulant MDS output for comparison with scalar.
    #[allow(dead_code)]
    pub fn mds_circulant_batch_8(states: &[[u64; 12]; 8]) -> [[u64; 12]; 8] {
        // AoS to SoA
        let mut packed = [unsafe { _mm512_set1_epi64(0) }; 12];
        for j in 0..12 {
            let mut vals = [0u64; 8];
            for i in 0..8 { vals[i] = states[i][j]; }
            packed[j] = unsafe { _mm512_loadu_si512(vals.as_ptr() as *const __m512i) };
        }
        let result_packed = mds_multiply_freq_packed(&packed);
        // SoA to AoS
        let mut result = [[0u64; 12]; 8];
        for j in 0..12 {
            let mut vals = [0u64; 8];
            unsafe { _mm512_storeu_si512(vals.as_mut_ptr() as *mut __m512i, result_packed[j]); }
            for i in 0..8 { result[i][j] = vals[i]; }
        }
        result
    }

    /// Batched hash_n_to_hash_no_pad for 8 inputs simultaneously.
    /// All 8 inputs MUST have the same length (required for lock-step processing).
    /// Returns 8 HashOut values (4 elements each).
    /// This processes the sponge absorb-permute loop in lock-step,
    /// keeping the state in packed SoA form between permute calls.
    #[inline]
    pub fn hash_batch_8(inputs: [&[u64]; 8]) -> [[u64; 4]; 8] {
        let len = inputs[0].len();
        let num_steps = (len + 7) / 8;

        if num_steps == 0 {
            // Empty inputs: permute zero state once
            let states = [[0u64; 12]; 8];
            let result = permute_batch_8(&states);
            let mut results = [[0u64; 4]; 8];
            for i in 0..8 {
                results[i] = [result[i][0], result[i][1], result[i][2], result[i][3]];
            }
            return results;
        }

        // Keep state in packed SoA form throughout the loop
        let mut packed: [P; 12] = [P::ZEROS; 12];

        for step in 0..num_steps {
            let start = step * 8;
            let end = (start + 8).min(len);
            let chunk_len = end - start;

            // Absorb: overwrite rate positions with input chunk
            // Directly update the packed state in SoA form
            // Only overwrite positions 0..chunk_len; leave the rest unchanged
            // Use direct load from each input slice
            for j in 0..chunk_len {
                let idx = start + j;
                // Load 8 values from inputs[0..8][idx] into a packed register
                // Safety: idx = start + j where j < chunk_len = min(8, len - start),
                // so idx < len. All inputs have the same length.
                let mut vals = [GoldilocksField::ZERO; 8];
                vals[0] = GoldilocksField(unsafe { *inputs[0].get_unchecked(idx) });
                vals[1] = GoldilocksField(unsafe { *inputs[1].get_unchecked(idx) });
                vals[2] = GoldilocksField(unsafe { *inputs[2].get_unchecked(idx) });
                vals[3] = GoldilocksField(unsafe { *inputs[3].get_unchecked(idx) });
                vals[4] = GoldilocksField(unsafe { *inputs[4].get_unchecked(idx) });
                vals[5] = GoldilocksField(unsafe { *inputs[5].get_unchecked(idx) });
                vals[6] = GoldilocksField(unsafe { *inputs[6].get_unchecked(idx) });
                vals[7] = GoldilocksField(unsafe { *inputs[7].get_unchecked(idx) });
                packed[j] = P(vals);
            }

            // Packed permute: process all 8 states in SoA form
            packed = permute_packed_8(&packed);
        }

        // Extract hash output (first 4 elements of each packed state)
        let mut results = [[0u64; 4]; 8];
        for j in 0..4 {
            for i in 0..8 {
                results[i][j] = packed[j].0[i].0;
            }
        }
        results
    }

    /// Batched compress for 8 pairs of hashes simultaneously.
    /// Each pair is (left_hash, right_hash), each 4 u64 elements.
    /// Returns 8 compressed HashOut values.
    #[inline]
    #[allow(dead_code)]
    pub fn compress_batch_8(
        left_hashes: &[[u64; 4]; 8],
        right_hashes: &[[u64; 4]; 8],
    ) -> [[u64; 4]; 8] {
        // Build 8 permutation states: state[0..4] = left, state[4..8] = right
        let mut states: [[u64; 12]; 8] = [[0; 12]; 8];
        for i in 0..8 {
            states[i][0] = left_hashes[i][0];
            states[i][1] = left_hashes[i][1];
            states[i][2] = left_hashes[i][2];
            states[i][3] = left_hashes[i][3];
            states[i][4] = right_hashes[i][0];
            states[i][5] = right_hashes[i][1];
            states[i][6] = right_hashes[i][2];
            states[i][7] = right_hashes[i][3];
            // state[8..12] = 0 (capacity)
        }

        // Batched permute
        states = permute_batch_8(&states);

        // Extract hash output (first 4 elements)
        let mut results = [[0u64; 4]; 8];
        for i in 0..8 {
            results[i] = [states[i][0], states[i][1], states[i][2], states[i][3]];
        }
        results
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    use crate::field::goldilocks_field::GoldilocksField as F;
    use crate::field::types::{Field, PrimeField64};
    use crate::hash::poseidon::test_helpers::{check_consistency, check_test_vectors};

    #[test]
    fn test_vectors() {
        // Test inputs are:
        // 1. all zeros
        // 2. range 0..WIDTH
        // 3. all -1's
        // 4. random elements of GoldilocksField.
        // expected output calculated with (modified) hadeshash reference implementation.

        let neg_one: u64 = F::NEG_ONE.to_canonical_u64();

        #[rustfmt::skip]
        let test_vectors12: Vec<([u64; 12], [u64; 12])> = vec![
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
             [0x3c18a9786cb0b359, 0xc4055e3364a246c3, 0x7953db0ab48808f4, 0xc71603f33a1144ca,
              0xd7709673896996dc, 0x46a84e87642f44ed, 0xd032648251ee0b3c, 0x1c687363b207df62,
              0xdf8565563e8045fe, 0x40f5b37ff4254dae, 0xd070f637b431067c, 0x1792b1c4342109d7, ]),
            ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ],
             [0xd64e1e3efc5b8e9e, 0x53666633020aaa47, 0xd40285597c6a8825, 0x613a4f81e81231d2,
              0x414754bfebd051f0, 0xcb1f8980294a023f, 0x6eb2a9e4d54a9d0f, 0x1902bc3af467e056,
              0xf045d5eafdc6021f, 0xe4150f77caaa3be5, 0xc9bfd01d39b50cce, 0x5c0a27fcb0e1459b, ]),
            ([neg_one, neg_one, neg_one, neg_one,
              neg_one, neg_one, neg_one, neg_one,
              neg_one, neg_one, neg_one, neg_one, ],
             [0xbe0085cfc57a8357, 0xd95af71847d05c09, 0xcf55a13d33c1c953, 0x95803a74f4530e82,
              0xfcd99eb30a135df1, 0xe095905e913a3029, 0xde0392461b42919b, 0x7d3260e24e81d031,
              0x10d3d0465d9deaa0, 0xa87571083dfc2a47, 0xe18263681e9958f8, 0xe28e96f1ae5e60d3, ]),
            ([0x8ccbbbea4fe5d2b7, 0xc2af59ee9ec49970, 0x90f7e1a9e658446a, 0xdcc0630a3ab8b1b8,
              0x7ff8256bca20588c, 0x5d99a7ca0c44ecfb, 0x48452b17a70fbee3, 0xeb09d654690b6c88,
              0x4a55d3a39c676a88, 0xc0407a38d2285139, 0xa234bac9356386d1, 0xe1633f2bad98a52f, ],
             [0xa89280105650c4ec, 0xab542d53860d12ed, 0x5704148e9ccab94f, 0xd3a826d4b62da9f5,
              0x8a7a6ca87892574f, 0xc7017e1cad1a674e, 0x1f06668922318e34, 0xa3b203bc8102676f,
              0xfcc781b0ce382bf2, 0x934c69ff3ed14ba5, 0x504688a5996e8f13, 0x401f3f2ed524a2ba, ]),
        ];

        check_test_vectors::<F>(test_vectors12);
    }

    #[test]
    fn consistency() {
        check_consistency::<F>();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[test]
    fn batched_permute_correctness() {
        use super::batched_poseidon::permute_batch_8;
        use super::GoldilocksField;
        use crate::field::types::Field;
        use crate::hash::poseidon::Poseidon;

        // Test with 8 different states
        let test_inputs: [[u64; 12]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            [0xFFFFFFFF00000001, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0; 12],
            [1; 12],
            [0xFFFFFFFF00000000; 12],
            [0x8ccbbbea4fe5d2b7, 0xc2af59ee9ec49970, 0x90f7e1a9e658446a, 0xdcc0630a3ab8b1b8,
             0x7ff8256bca20588c, 0x5d99a7ca0c44ecfb, 0x48452b17a70fbee3, 0xeb09d654690b6c88,
             0x4a55d3a39c676a88, 0xc0407a38d2285139, 0xa234bac9356386d1, 0xe1633f2bad98a52f],
            [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0xDEADBEEFCAFEBABE, 0x13579BDF2468ACE0,
             0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444,
             0x5555555555555555, 0x6666666666666666, 0x7777777777777777, 0x8888888888888888],
        ];

        let batch_result = permute_batch_8(&test_inputs);

        // Compare with scalar permutation for each state
        for i in 0..8 {
            let state: [GoldilocksField; 12] = core::array::from_fn(|j| GoldilocksField(test_inputs[i][j]));
            let scalar_result = <GoldilocksField as Poseidon>::poseidon(state);

            for j in 0..12 {
                assert_eq!(batch_result[i][j], scalar_result[j].0,
                    "Mismatch at state {} element {}: batch={:016x} scalar={:016x}",
                    i, j, batch_result[i][j], scalar_result[j].0);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[test]
    fn batched_hash_correctness() {
        use super::batched_poseidon::hash_batch_8;
        use super::GoldilocksField;
        use crate::hash::hash_types::{HashOut, RichField};
        use crate::hash::hashing::hash_n_to_hash_no_pad;
        use crate::hash::poseidon::{PoseidonHash, PoseidonPermutation};
        use crate::plonk::config::Hasher;

        // Test with 8 inputs of the same length (required for lock-step batching)
        let input_8_elems: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let input_16_elems: [u64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let input_9_elems: [u64; 9] = [100, 200, 300, 400, 500, 600, 700, 800, 900];

        // Test case 1: 8 elements each (1 absorb step)
        {
            let inputs: [&[u64]; 8] = [&input_8_elems; 8];
            let batch_results = hash_batch_8(inputs);
            let scalar_hash = <PoseidonHash as Hasher<GoldilocksField>>::hash_no_pad(
                &input_8_elems.map(GoldilocksField)
            );
            for i in 0..8 {
                for j in 0..4 {
                    assert_eq!(batch_results[i][j], scalar_hash.elements[j].0,
                        "Mismatch 8-elem input {} element {}: batch={:016x} scalar={:016x}",
                        i, j, batch_results[i][j], scalar_hash.elements[j].0);
                }
            }
        }

        // Test case 2: 16 elements each (2 absorb steps)
        {
            let inputs: [&[u64]; 8] = [&input_16_elems; 8];
            let batch_results = hash_batch_8(inputs);
            let scalar_hash = <PoseidonHash as Hasher<GoldilocksField>>::hash_no_pad(
                &input_16_elems.map(GoldilocksField)
            );
            for i in 0..8 {
                for j in 0..4 {
                    assert_eq!(batch_results[i][j], scalar_hash.elements[j].0,
                        "Mismatch 16-elem input {} element {}", i, j);
                }
            }
        }

        // Test case 3: 9 elements each (2 absorb steps, partial second)
        {
            let inputs: [&[u64]; 8] = [&input_9_elems; 8];
            let batch_results = hash_batch_8(inputs);
            let scalar_hash = <PoseidonHash as Hasher<GoldilocksField>>::hash_no_pad(
                &input_9_elems.map(GoldilocksField)
            );
            for i in 0..8 {
                for j in 0..4 {
                    assert_eq!(batch_results[i][j], scalar_hash.elements[j].0,
                        "Mismatch 9-elem input {} element {}", i, j);
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[test]
    fn batched_compress_correctness() {
        use super::batched_poseidon::compress_batch_8;
        use super::GoldilocksField;
        use crate::hash::hash_types::HashOut;
        use crate::hash::hashing::compress;
        use crate::hash::poseidon::PoseidonPermutation;

        // Test with 8 different pairs
        let left: [[u64; 4]; 8] = [
            [1, 2, 3, 4],
            [100, 200, 300, 400],
            [0xFFFFFFFF00000001, 2, 3, 4],
            [0; 4],
            [1; 4],
            [0xFFFFFFFF00000000; 4],
            [0x8ccbbbea4fe5d2b7, 0xc2af59ee9ec49970, 0x90f7e1a9e658446a, 0xdcc0630a3ab8b1b8],
            [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0xDEADBEEFCAFEBABE, 0x13579BDF2468ACE0],
        ];
        let right: [[u64; 4]; 8] = [
            [5, 6, 7, 8],
            [500, 600, 700, 800],
            [5, 6, 7, 8],
            [1; 4],
            [2; 4],
            [0x123456789ABCDEF0; 4],
            [0x7ff8256bca20588c, 0x5d99a7ca0c44ecfb, 0x48452b17a70fbee3, 0xeb09d654690b6c88],
            [0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444],
        ];

        let batch_results = compress_batch_8(&left, &right);

        // Compare with scalar compress for each pair
        for i in 0..8 {
            let l = HashOut { elements: [GoldilocksField(left[i][0]), GoldilocksField(left[i][1]), GoldilocksField(left[i][2]), GoldilocksField(left[i][3])] };
            let r = HashOut { elements: [GoldilocksField(right[i][0]), GoldilocksField(right[i][1]), GoldilocksField(right[i][2]), GoldilocksField(right[i][3])] };
            let scalar_result = compress::<GoldilocksField, PoseidonPermutation<GoldilocksField>>(l, r);

            for j in 0..4 {
                assert_eq!(batch_results[i][j], scalar_result.elements[j].0,
                    "Mismatch at pair {} element {}: batch={:016x} scalar={:016x}",
                    i, j, batch_results[i][j], scalar_result.elements[j].0);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[test]
    fn batched_mds_layer_correctness() {
        use super::batched_poseidon::mds_layer_batch;
        use super::GoldilocksField;
        use crate::hash::poseidon::Poseidon;
        use crate::field::types::Field;
        use plonky2_field::arch::x86_64::avx512_goldilocks_field::Avx512GoldilocksField as P;
        use plonky2_field::packed::PackedField;

        // Test with 8 different states
        let test_inputs: [[u64; 12]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            [0xFFFFFFFF00000001, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0; 12],
            [1; 12],
            [0xFFFFFFFF00000000; 12],
            [0x8ccbbbea4fe5d2b7, 0xc2af59ee9ec49970, 0x90f7e1a9e658446a, 0xdcc0630a3ab8b1b8,
             0x7ff8256bca20588c, 0x5d99a7ca0c44ecfb, 0x48452b17a70fbee3, 0xeb09d654690b6c88,
             0x4a55d3a39c676a88, 0xc0407a38d2285139, 0xa234bac9356386d1, 0xe1633f2bad98a52f],
            [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0xDEADBEEFCAFEBABE, 0x13579BDF2468ACE0,
             0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444,
             0x5555555555555555, 0x6666666666666666, 0x7777777777777777, 0x8888888888888888],
        ];

        // Convert to SoA packed format
        let mut packed = [P::ZEROS; 12];
        for j in 0..12 {
            let mut vals = [GoldilocksField::ZERO; 8];
            for i in 0..8 {
                vals[i] = GoldilocksField(test_inputs[i][j]);
            }
            packed[j] = P(vals);
        }

        let batch_result = mds_layer_batch(&packed);

        // Compare with scalar MDS for each state
        for i in 0..8 {
            let state: [GoldilocksField; 12] = core::array::from_fn(|j| GoldilocksField(test_inputs[i][j]));
            let scalar_result = GoldilocksField::mds_layer(&state);

            for j in 0..12 {
                let batch_val = batch_result[j].0[i];
                assert_eq!(batch_val, scalar_result[j],
                    "Mismatch at state {} element {}: batch={:?} scalar={:?}",
                    i, j, batch_val, scalar_result[j]);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw", target_feature = "avx512dq", target_feature = "avx512vl"))]
    #[test]
    fn batched_mds_correctness() {
        use super::batched_poseidon::mds_circulant_batch_8;
        use crate::hash::poseidon_goldilocks::poseidon12_mds;
        use crate::field::types::Field;

        // Test with 8 different states
        let test_states: [[u64; 12]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            [0xFFFFFFFF00000001, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0; 12],
            [1; 12],
            [0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000,
             0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000,
             0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000, 0xFFFFFFFF00000000],
            [0x8ccbbbea4fe5d2b7, 0xc2af59ee9ec49970, 0x90f7e1a9e658446a, 0xdcc0630a3ab8b1b8,
             0x7ff8256bca20588c, 0x5d99a7ca0c44ecfb, 0x48452b17a70fbee3, 0xeb09d654690b6c88,
             0x4a55d3a39c676a88, 0xc0407a38d2285139, 0xa234bac9356386d1, 0xe1633f2bad98a52f],
            [0x123456789ABCDEF0, 0xFEDCBA9876543210, 0xDEADBEEFCAFEBABE, 0x13579BDF2468ACE0,
             0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444,
             0x5555555555555555, 0x6666666666666666, 0x7777777777777777, 0x8888888888888888],
        ];

        let batch_result = mds_circulant_batch_8(&test_states);

        for i in 0..8 {
            // Split into lo/hi and apply scalar mds_multiply_freq to each half
            let mut lo = [0u64; 12];
            let mut hi = [0u64; 12];
            for r in 0..12 {
                hi[r] = test_states[i][r] >> 32;
                lo[r] = (test_states[i][r] as u32) as u64;
            }
            let scalar_lo = poseidon12_mds::mds_multiply_freq(lo);
            let scalar_hi = poseidon12_mds::mds_multiply_freq(hi);

            // Combine
            let mut expected = [0u64; 12];
            for r in 0..12 {
                let s = scalar_lo[r] as u128 + ((scalar_hi[r] as u128) << 32);
                expected[r] = s as u64; // Low 64 bits (the non-canonical form before reduction)
            }

            assert_eq!(batch_result[i], expected, "Mismatch at state {}", i);
        }
    }
}
