package v2transport

import (
	"bytes"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/pearl-research-labs/pearl/node/btcec"
	"github.com/pearl-research-labs/pearl/node/btcec/ellswift"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setHex(hexString string) *btcec.FieldVal {
	if len(hexString)%2 != 0 {
		hexString = "0" + hexString
	}
	bytes, _ := hex.DecodeString(hexString)

	var f btcec.FieldVal
	f.SetByteSlice(bytes)

	return &f
}

const (
	mainNet = 0xd9b4bef9
)

func TestPacketEncodingVectors(t *testing.T) {
	tests := []struct {
		inIdx                 int
		inPrivOurs            string
		inEllswiftOurs        string
		inEllswiftTheirs      string
		inInitiating          bool
		inContents            string
		inMultiply            int
		inAad                 string
		inIgnore              bool
		midXOurs              string
		midXTheirs            string
		midXShared            string
		midSharedSecret       string
		midInitiatorL         string
		midInitiatorP         string
		midResponderL         string
		midResponderP         string
		midSendGarbageTerm    string
		midRecvGarbageTerm    string
		outSessionID          string
		outCiphertext         string
		outCiphertextEndsWith string
	}{
		{
			inIdx:                 1,
			inPrivOurs:            "61062ea5071d800bbfd59e2e8b53d47d194b095ae5a4df04936b49772ef0d4d7",
			inEllswiftOurs:        "ec0adff257bbfe500c188c80b4fdd640f6b45a482bbc15fc7cef5931deff0aa186f6eb9bba7b85dc4dcc28b28722de1e3d9108b985e2967045668f66098e475b",
			inEllswiftTheirs:      "a4a94dfce69b4a2a0a099313d10f9f7e7d649d60501c9e1d274c300e0d89aafaffffffffffffffffffffffffffffffffffffffffffffffffffffffff8faf88d5",
			inInitiating:          true,
			inContents:            "8e",
			inMultiply:            1,
			inAad:                 "",
			inIgnore:              false,
			midXOurs:              "19e965bc20fc40614e33f2f82d4eeff81b5e7516b12a5c6c0d6053527eba0923",
			midXTheirs:            "0c71defa3fafd74cb835102acd81490963f6b72d889495e06561375bd65f6ffc",
			midXShared:            "4eb2bf85bd00939468ea2abb25b63bc642e3d1eb8b967fb90caa2d89e716050e",
			midSharedSecret:       "c6992a117f5edbea70c3f511d32d26b9798be4b81a62eaee1a5acaa8459a3592",
			midInitiatorL:         "8bc4cbc0bdf7d4d41c37f820cd8600c0eacc3752749dda51e6196cde4f388f3f",
			midInitiatorP:         "97f03f63c239dc78f9617992b31bdccb95baac8a0ff20b5deeebd061efb9541f",
			midResponderL:         "b940c804d45bb3e7bdf9dd4bb13cd2ac2693cb2ac06916f8287d762b0b551362",
			midResponderP:         "2a40174e52a9c20c8ee54ee2cf5f69114569c6bf1db34866ecbb53bb64cb529c",
			midSendGarbageTerm:    "a4f1fbbc7724fe0efd75775a7c5c3b04",
			midRecvGarbageTerm:    "5d8aa14d72ffd5618eee99608bd090b0",
			outSessionID:          "227655577bae2bd7f4a6700e7a01e60362a1e584a034ad331df4052023d76f56",
			outCiphertext:         "deb867d0561edab03568c84eafa18fc15230287f55",
			outCiphertextEndsWith: "",
		},
		{
			inIdx:                 999,
			inPrivOurs:            "1f9c581b35231838f0f17cf0c979835baccb7f3abbbb96ffcc318ab71e6e126f",
			inEllswiftOurs:        "a1855e10e94e00baa23041d916e259f7044e491da6171269694763f018c7e63693d29575dcb464ac816baa1be353ba12e3876cba7628bd0bd8e755e721eb0140",
			inEllswiftTheirs:      "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f0000000000000000000000000000000000000000000000000000000000000000",
			inInitiating:          false,
			inContents:            "3eb1d4e98035cfd8eeb29bac969ed3824a",
			inMultiply:            1,
			inAad:                 "",
			inIgnore:              false,
			midXOurs:              "45b6f1f684fd9f2b16e2651ddc47156c0695c8c5cd2c0c9df6d79a1056c61120",
			midXTheirs:            "edd1fd3e327ce90cc7a3542614289aee9682003e9cf7dcc9cf2ca9743be5aa0c",
			midXShared:            "c40eb6190caf399c9007254ad5e5fa20d64af2b41696599c59b2191d16992955",
			midSharedSecret:       "a0138f564f74d0ad70bc337dacc9d0bf1d2349364caf1188a1e6e8ddb3b7b184",
			midInitiatorL:         "32dc5929d3c78e26c049ba4076a858ce48ca4b8d74cacca9e86388ebde0949ec",
			midInitiatorP:         "71c2a1d2dba32a8c0d69f8806e8433d15feb349fbf1fd89d0bd46a1848b9d503",
			midResponderL:         "c0bfaed56bbb7baa2ca237dab55cd35eca0cf605881ccfa9ab41113e8d2bf34f",
			midResponderP:         "57d0d565cdef58b8daddf891e35ff2500fb86fc8e4ca1f6f0fefe34fd2c93c35",
			midSendGarbageTerm:    "c012bb4fc45cd9189b790d5a577b0743",
			midRecvGarbageTerm:    "303917ebd089ee59f90f7f81b80210f3",
			outSessionID:          "f49261a242af8bcd8b596068b3574e605562b5782e0dddaddad2c3f6d2dd4787",
			outCiphertext:         "464ecdda4903021058b7d8a8401c1dade5a3db9347c6889dc2259706c6c3eeec03f924f268",
			outCiphertextEndsWith: "",
		},
		{
			inIdx:                 0,
			inPrivOurs:            "0286c41cd30913db0fdff7a64ebda5c8e3e7cef10f2aebc00a7650443cf4c60d",
			inEllswiftOurs:        "d1ee8a93a01130cbf299249a258f94feb5f469e7d0f2f28f69ee5e9aa8f9b54a60f2c3ff2d023634ec7f4127a96cc11662e402894cf1f694fb9a7eaa5f1d9244",
			inEllswiftTheirs:      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff22d5e441524d571a52b3def126189d3f416890a99d4da6ede2b0cde1760ce2c3f98457ae",
			inInitiating:          true,
			inContents:            "054290a6c6ba8d80478172e89d32bf690913ae9835de6dcf206ff1f4d652286fe0ddf74deba41d55de3edc77c42a32af79bbea2c00bae7492264c60866ae5a",
			inMultiply:            1,
			inAad:                 "84932a55aac22b51e7b128d31d9f0550da28e6a3f394224707d878603386b2f9d0c6bcd8046679bfed7b68c517e7431e75d9dd34605727d2ef1c2babbf680ecc8d68d2c4886e9953a4034abde6da4189cd47c6bb3192242cf714d502ca6103ee84e08bc2ca4fd370d5ad4e7d06c7fbf496c6c7cc7eb19c40c61fb33df2a9ba48497a96c98d7b10c1f91098a6b7b16b4bab9687f27585ade1491ae0dba6a79e1e2d85dd9d9d45c5135ca5fca3f0f99a60ea39edbc9efc7923111c937913f225d67788d5f7e8852b697e26b92ec7bfcaa334a1665511c2b4c0a42d06f7ab98a9719516c8fd17f73804555ee84ab3b7d1762f6096b778d3cb9c799cbd49a9e4a325197b4e6cc4a5c4651f8b41ff88a92ec428354531f970263b467c77ed11312e2617d0d53fe9a8707f51f9f57a77bfb49afe3d89d85ec05ee17b9186f360c94ab8bb2926b65ca99dae1d6ee1af96cad09de70b6767e949023e4b380e66669914a741ed0fa420a48dbc7bfae5ef2019af36d1022283dd90655f25eec7151d471265d22a6d3f91dc700ba749bb67c0fe4bc0888593fbaf59d3c6fff1bf756a125910a63b9682b597c20f560ecb99c11a92c8c8c3f7fbfaa103146083a0ccaecf7a5f5e735a784a8820155914a289d57d8141870ffcaf588882332e0bcd8779efa931aa108dab6c3cce76691e345df4a91a03b71074d66333fd3591bff071ea099360f787bbe43b7b3dff2a59c41c7642eb79870222ad1c6f2e5a191ed5acea51134679587c9cf71c7d8ee290be6bf465c4ee47897a125708704ad610d8d00252d01959209d7cd04d5ecbbb1419a7e84037a55fefa13dee464b48a35c96bcb9a53e7ed461c3a1607ee00c3c302fd47cd73fda7493e947c9834a92d63dcfbd65aa7c38c3e3a2748bb5d9a58e7495d243d6b741078c8f7ee9c8813e473a323375702702b0afae1550c8341eedf5247627343a95240cb02e3e17d5dca16f8d8d3b2228e19c06399f8ec5c5e9dbe4caef6a0ea3ffb1d3c7eac03ae030e791fa12e537c80d56b55b764cadf27a8701052df1282ba8b5e3eb62b5dc7973ac40160e00722fa958d95102fc25c549d8c0e84bed95b7acb61ba65700c4de4feebf78d13b9682c52e937d23026fb4c6193e6644e2d3c99f91f4f39a8b9fc6d013f89c3793ef703987954dc0412b550652c01d922f525704d32d70d6d4079bc3551b563fb29577b3aecdc9505011701dddfd94830431e7a4918927ee44fb3831ce8c4513839e2deea1287f3fa1ab9b61a256c09637dbc7b4f0f8fbb783840f9c24526da883b0df0c473cf231656bd7bc1aaba7f321fec0971c8c2c3444bff2f55e1df7fea66ec3e440a612db9aa87bb505163a59e06b96d46f50d8120b92814ac5ab146bc78dbbf91065af26107815678ce6e33812e6bf3285d4ef3b7b04b076f21e7820dcbfdb4ad5218cf4ff6a65812d8fcb98ecc1e95e2fa58e3efe4ce26cd0bd400d6036ab2ad4f6c713082b5e3f1e04eb9e3b6c8f63f57953894b9e220e0130308e1fd91f72d398c1e7962ca2c31be83f31d6157633581a0a6910496de8d55d3d07090b6aa087159e388b7e7dec60f5d8a60d93ca2ae91296bd484d916bfaaa17c8f45ea4b1a91b37c82821199a2b7596672c37156d8701e7352aa48671d3b1bbbd2bd5f0a2268894a25b0cb2514af39c8743f8cce8ab4b523053739fd8a522222a09acf51ac704489cf17e4b7125455cb8f125b4d31af1eba1f8cf7f81a5a100a141a7ee72e8083e065616649c241f233645c5fc865d17f0285f5c52d9f45312c979bfb3ce5f2a1b951deddf280ffb3f370410cffd1583bfa90077835aa201a0712d1dcd1293ee177738b14e6b5e2a496d05220c3253bb6578d6aff774be91946a614dd7e879fb3dcf7451e0b9adb6a8c44f53c2c464bcc0019e9fad89cac7791a0a3f2974f759a9856351d4d2d7c5612c17cfc50f8479945df57716767b120a590f4bf656f4645029a525694d8a238446c5f5c2c1c995c09c1405b8b1eb9e0352ffdf766cc964f8dcf9f8f043dfab6d102cf4b298021abd78f1d9025fa1f8e1d710b38d9d1652f2d88d1305874ec41609b6617b65c5adb19b6295dc5c5da5fdf69f28144ea12f17c3c6fcce6b9b5157b3dfc969d6725fa5b098a4d9b1d31547ed4c9187452d281d0a5d456008caf1aa251fac8f950ca561982dc2dc908d3691ee3b6ad3ae3d22d002577264ca8e49c523bd51c4846be0d198ad9407bf6f7b82c79893eb2c05fe9981f687a97a4f01fe45ff8c8b7ecc551135cd960a0d6001ad35020be07ffb53cb9e731522ca8ae9364628914b9b8e8cc2f37f03393263603cc2b45295767eb0aac29b0930390eb89587ab2779d2e3decb8042acece725ba42eda650863f418f8d0d50d104e44fbbe5aa7389a4a144a8cecf00f45fb14c39112f9bfb56c0acbd44fa3ff261f5ce4acaa5134c2c1d0cca447040820c81ab1bcdc16aa075b7c68b10d06bbb7ce08b5b805e0238f24402cf24a4b4e00701935a0c68add3de090903f9b85b153cb179a582f57113bfc21c2093803f0cfa4d9d4672c2b05a24f7e4c34a8e9101b70303a7378b9c50b6cddd46814ef7fd73ef6923feceab8fc5aa8b0d185f2e83c7a99dcb1077c0ab5c1f5d5f01ba2f0420443f75c4417db9ebf1665efbb33dca224989920a64b44dc26f682cc77b4632c8454d49135e52503da855bc0f6ff8edc1145451a9772c06891f41064036b66c3119a0fc6e80dffeb65dc456108b7ca0296f4175fff3ed2b0f842cd46bd7e86f4c62dfaf1ddbf836263c00b34803de164983d0811cebfac86e7720c726d3048934c36c23189b02386a722ca9f0fe00233ab50db928d3bccea355cc681144b8b7edcaae4884d5a8f04425c0890ae2c74326e138066d8c05f4c82b29df99b034ea727afde590a1f2177ace3af99cfb1729d6539ce7f7f7314b046aab74497e63dd399e1f7d5f16517c23bd830d1fdee810f3c3b77573dd69c4b97d80d71fb5a632e00acdfa4f8e829faf3580d6a72c40b28a82172f8dcd4627663ebf6069736f21735fd84a226f427cd06bb055f94e7c92f31c48075a2955d82a5b9d2d0198ce0d4e131a112570a8ee40fb80462a81436a58e7db4e34b6e2c422e82f934ecda9949893da5730fc5c23c7c920f363f85ab28cc6a4206713c3152669b47efa8238fa826735f17b4e78750276162024ec85458cd5808e06f40dd9fd43775a456a3ff6cae90550d76d8b2899e0762ad9a371482b3e38083b1274708301d6346c22fea9bb4b73db490ff3ab05b2f7f9e187adef139a7794454b7300b8cc64d3ad76c0e4bc54e08833a4419251550655380d675bc91855aeb82585220bb97f03e976579c08f321b5f8f70988d3061f41465517d53ac571dbf1b24b94443d2e9a8e8a79b392b3d6a4ecdd7f626925c365ef6221305105ce9b5f5b6ecc5bed3d702bd4b7f5008aa8eb8c7aa3ade8ecf6251516fbefeea4e1082aa0e1848eddb31ffe44b04792d296054402826e4bd054e671f223e5557e4c94f89ca01c25c44f1a2ff2c05a70b43408250705e1b858bf0670679fdcd379203e36be3500dd981b1a6422c3cf15224f7fefdef0a5f225c5a09d15767598ecd9e262460bb33a4b5d09a64591efabc57c923d3be406979032ae0bc0997b65336a06dd75b253332ad6a8b63ef043f780a1b3fb6d0b6cad98b1ef4a02535eb39e14a866cfc5fc3a9c5deb2261300d71280ebe66a0776a151469551c3c5fa308757f956655278ec6330ae9e3625468c5f87e02cd9a6489910d4143c1f4ee13aa21a6859d907b788e28572fecee273d44e4a900fa0aa668dd861a60fb6b6b12c2c5ef3c8df1bd7ef5d4b0d1cdb8c15fffbb365b9784bd94abd001c6966216b9b67554ad7cb7f958b70092514f7800fc40244003e0fd1133a9b850fb17f4fcafde07fc87b07fb510670654a5d2d6fc9876ac74728ea41593beef003d6858786a52d3a40af7529596767c17000bfaf8dc52e871359f4ad8bf6e7b2853e5229bdf39657e213580294a5317c5df172865e1e17fe37093b585e04613f5f078f761b2b1752eb32983afda24b523af8851df9a02b37e77f543f18888a782a994a50563334282bf9cdfccc183fdf4fcd75ad86ee0d94f91ee2300a5befbccd14e03a77fc031a8cfe4f01e4c5290f5ac1da0d58ea054bd4837cfd93e5e34fc0eb16e48044ba76131f228d16cde9b0bb978ca7cdcd10653c358bdb26fdb723a530232c32ae0a4cecc06082f46e1c1d596bfe60621ad1e354e01e07b040cc7347c016653f44d926d13ca74e6cbc9d4ab4c99f4491c95c76fff5076b3936eb9d0a286b97c035ca88a3c6309f5febfd4cdaac869e4f58ed409b1e9eb4192fb2f9c2f12176d460fd98286c9d6df84598f260119fd29c63f800c07d8df83d5cc95f8c2fea2812e7890e8a0718bb1e031ecbebc0436dcf3e3b9a58bcc06b4c17f711f80fe1dffc3326a6eb6e00283055c6dabe20d311bfd5019591b7954f8163c9afad9ef8390a38f3582e0a79cdf0353de8eeb6b5f9f27b16ffdef7dd62869b4840ee226ccdce95e02c4545eb981b60571cd83f03dc5eaf8c97a0829a4318a9b3dc06c0e003db700b2260ff1fa8fee66890e637b109abb03ec901b05ca599775f48af50154c0e67d82bf0f558d7d3e0778dc38bea1eb5f74dc8d7f90abdf5511a424be66bf8b6a3cacb477d2e7ef4db68d2eba4d5289122d851f9501ba7e9c4957d8eba3be3fc8e785c4265a1d65c46f2809b70846c693864b169c9dcb78be26ea14b8613f145b01887222979a9e67aee5f800caa6f5c4229bdeefc901232ace6143c9865e4d9c07f51aa200afaf7e48a7d1d8faf366023beab12906ffcb3eaf72c0eb68075e4daf3c080e0c31911befc16f0cc4a09908bb7c1e26abab38bd7b788e1a09c0edf1a35a38d2ff1d3ed47fcdaae2f0934224694f5b56705b9409b6d3d64f3833b686f7576ec64bbdd6ff174e56c2d1edac0011f904681a73face26573fbba4e34652f7ae84acfb2fa5a5b3046f98178cd0831df7477de70e06a4c00e305f31aafc026ef064dd68fd3e4252b1b91d617b26c6d09b6891a00df68f105b5962e7f9d82da101dd595d286da721443b72b2aba2377f6e7772e33b3a5e3753da9c2578c5d1daab80187f55518c72a64ee150a7cb5649823c08c9f62cd7d020b45ec2cba8310db1a7785a46ab24785b4d54ff1660b5ca78e05a9a55edba9c60bf044737bc468101c4e8bd1480d749be5024adefca1d998abe33eaeb6b11fbb39da5d905fdd3f611b2e51517ccee4b8af72c2d948573505590d61a6783ab7278fc43fe55b1fcc0e7216444d3c8039bb8145ef1ce01c50e95a3f3feab0aee883fdb94cc13ee4d21c542aa795e18932228981690f4d4c57ca4db6eb5c092e29d8a05139d509a8aeb48baa1eb97a76e597a32b280b5e9d6c36859064c98ff96ef5126130264fa8d2f49213870d9fb036cff95da51f270311d9976208554e48ffd486470d0ecdb4e619ccbd8226147204baf8e235f54d8b1cba8fa34a9a4d055de515cdf180d2bb6739a175183c472e30b5c914d09eeb1b7dafd6872b38b48c6afc146101200e6e6a44fe5684e220adc11f5c403ddb15df8051e6bdef09117a3a5349938513776286473a3cf1d2788bb875052a2e6459fa7926da33380149c7f98d7700528a60c954e6f5ecb65842fde69d614be69eaa2040a4819ae6e756accf936e14c1e894489744a79c1f2c1eb295d13e2d767c09964b61f9cfe497649f712",
			inIgnore:              false,
			midXOurs:              "33a32d10066fa3963a9518a14d1bd1cb5ccaceaeaaeddb4d7aead90c08395bfd",
			midXTheirs:            "568146140669e69646a6ffeb3793e8010e2732209b4c34ec13e209a070109183",
			midXShared:            "a1017beaa8784f283dee185cd847ae3a327a981e62ae21e8c5face175fc97e9b",
			midSharedSecret:       "250b93570d411149105ab8cb0bc5079914906306368c23e9d77c2a33265b994c",
			midInitiatorL:         "5b98143b3aee4ee2f66d58f2c1fcc80babcc95ec14436b70b5c19a5a9a4297fb",
			midInitiatorP:         "ae0d2426ebacdcc4f4b3c7114f16a5fd1fbad0ad74f2def96058431c7b13079a",
			midResponderL:         "226e583fc07b17d8dd03abadf83f286123ff9e41a9bee4a9c49e94f4c36c5fd1",
			midResponderP:         "82c761882bc7d4f83cf496f381346bfb84edaa60bb03c25f388d94f9890d93af",
			midSendGarbageTerm:    "8dcd8b02de2e44cf9323339173eafd57",
			midRecvGarbageTerm:    "cdb0462d997f2fb79b08f44c2fe7d1d9",
			outSessionID:          "26a36bde362244a30bf6e26cbba641385df610bf403aa09e31a9f45828e4521d",
			outCiphertext:         "262e6c4d1c086a59f339fa087f765987f0f45a538e530c626a62d78b4beb37626a937bba5f4b9ba5e112cd65138b8f66953287266108c528c036e5e61a3798144bfff3acac22c3547ca4c017ed84e717392945",
			outCiphertextEndsWith: "",
		},
		{
			inIdx:                 223,
			inPrivOurs:            "6c77432d1fda31e9f942f8af44607e10f3ad38a65f8a4bddae823e5eff90dc38",
			inEllswiftOurs:        "d2685070c1e6376e633e825296634fd461fa9e5bdf2109bcebd735e5a91f3e587c5cb782abb797fbf6bb5074fd1542a474f2a45b673763ec2db7fb99b737bbb9",
			inEllswiftTheirs:      "56bd0c06f10352c3a1a9f4b4c92f6fa2b26df124b57878353c1fc691c51abea77c8817daeeb9fa546b77c8daf79d89b22b0e1b87574ece42371f00237aa9d83a",
			inInitiating:          false,
			inContents:            "7e0e78eb6990b059e6cf0ded66ea93ef82e72aa2f18ac24f2fc6ebab561ae557420729da103f64cecfa20527e15f9fb669a49bbbf274ef0389b3e43c8c44e5f60bf2ac38e2b55e7ec4273dba15ba41d21f8f5b3ee1688b3c29951218caf847a97fb50d75a86515d445699497d968164bf740012679b8962de573be941c62b7ef",
			inMultiply:            1,
			inAad:                 "",
			inIgnore:              true,
			midXOurs:              "193d019db571162e52567e0cfdf9dd6964394f32769ae2edc4933b03b502d771",
			midXTheirs:            "2dd7b9cc85524f8670f695c3143ac26b45cebcabb2782a85e0fe15aee3956535",
			midXShared:            "5e35f94adfd57976833bffec48ef6dde983d18a55501154191ea352ef06732ee",
			midSharedSecret:       "1918b741ef5f9d1d7670b050c152b4a4ead2c31be9aecb0681c0cd4324150853",
			midInitiatorL:         "17085c627e3cb25b71e17875f02ddff535270eab649c11ce61908eff6db540f9",
			midInitiatorP:         "f7ed4fbde294c393866ef74b8a0055dffa3bebc243804ed576eca5e88af3cb9e",
			midResponderL:         "3f5238d68ad29379cc5a2203738351d63a798374291bc5bf4bf9f00a088c1bcc",
			midResponderP:         "72256960ab3e77062c62021aeaa7900e20b634bde8a443cfaee42677aad8849f",
			midSendGarbageTerm:    "7a78a182e453c02e1cd4b69a4f225ed4",
			midRecvGarbageTerm:    "cea60ce548d3606db36bcb809e799ff6",
			outSessionID:          "6e72adaae3ad478ff961fc68047aef673e3cbf95f2a58bd507b14728fd133b83",
			outCiphertext:         "",
			outCiphertextEndsWith: "28645a274c6c78b32fd499d7c2a6ea46ba563bc4f95f99d84491f56bd1abbcfc",
		},
		{
			inIdx:                 448,
			inPrivOurs:            "a6ec25127ca1aa4cf16b20084ba1e6516baae4d32422288e9b36d8bddd2de35a",
			inEllswiftOurs:        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff053d7ecca53e33e185a8b9be4e7699a97c6ff4c795522e5918ab7cd6b6884f67e683f3dc",
			inEllswiftTheirs:      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffa7730be30000000000000000000000000000000000000000000000000000000000000000",
			inInitiating:          true,
			inContents:            "00cf68f8f7ac49ffaa02c4864fdf6dfe7bbf2c740b88d98c50ebafe32c92f3427f57601ffcb21a3435979287db8fee6c302926741f9d5e464c647eeb9b7acaeda46e00abd7506fc9a719847e9a7328215801e96198dac141a15c7c2f68e0690dd1176292a0dded04d1f548aad88f1aebdc0a8f87da4bb22df32dd7c160c225b843e83f6525d6d484f502f16d923124fc538794e21da2eb689d18d87406ecced5b9f92137239ed1d37bcfa7836641a83cf5e0a1cf63f51b06f158e499a459ede41c",
			inMultiply:            1,
			inAad:                 "",
			inIgnore:              false,
			midXOurs:              "02b225089255f7b02b20276cfe9779144df8fb1957b477bff3239d802d1256e9",
			midXTheirs:            "5232c4b6bde9d3d45d7b763ebd7495399bb825cc21de51011761cd81a51bdc84",
			midXShared:            "379223d2f1ea7f8a22043c4ce4122623098309e15b1ce58286ebe3d3bf40f4e1",
			midSharedSecret:       "dd210aa6629f20bb328e5d89daa6eb2ac3d1c658a725536ff154f31b536c23b2",
			midInitiatorL:         "261d01a737f475e2d67845088e4bbb6d0da58e52e7fdf4559816ad9a3b6afe21",
			midInitiatorP:         "9deed74aeab6c18d99d69c6bc451c826792b9eab8ac6c3a74b95121cc0041127",
			midResponderL:         "dfb599d4574f234a848f246bec11087f55e53e28c2178c235b2f639ea19d40a9",
			midResponderP:         "9b53615fff5f78330bb79da5f42744ff13826953500e962c11dce0ebf8917f21",
			midSendGarbageTerm:    "bbab0ac8bd00bb7a295fb9842273c883",
			midRecvGarbageTerm:    "6aa075e5aca8729a7b2207733fbb101b",
			outSessionID:          "7bdb716da9fab1931ac8a466c778c4a33acc24956a84ca67ad6ad970e117572d",
			outCiphertext:         "",
			outCiphertextEndsWith: "21ae92a92030b80ace8749ad965d2a5c9560f0cede2cc586889a680496bde4c1",
		},
		{
			inIdx:                 673,
			inPrivOurs:            "0af952659ed76f80f585966b95ab6e6fd68654672827878684c8b547b1b94f5a",
			inEllswiftOurs:        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffc81017fd92fd31637c26c906b42092e11cc0d3afae8d9019d2578af22735ce7bc469c72d",
			inEllswiftTheirs:      "9652d78baefc028cd37a6a92625b8b8f85fde1e4c944ad3f20e198bef8c02f19fffffffffffffffffffffffffffffffffffffffffffffffffffffffff2e91870",
			inInitiating:          false,
			inContents:            "5c6272ee55da855bbbf7b1246d9885aa7aa601a715ab86fa46c50da533badf82b97597c968293ae04e",
			inMultiply:            97561,
			inAad:                 "",
			inIgnore:              false,
			midXOurs:              "4b1767466fe2fb8deddf2dc52cc19c7e2032007e19bfb420b30a80152d0f22d6",
			midXTheirs:            "64c383e0e78ac99476ddff2061683eeefa505e3666673a1371342c3e6c26981d",
			midXShared:            "5bcfeac98d87e87e158bf839f1269705429f7af2a25b566a25811b5f9aef9560",
			midSharedSecret:       "3568f2aea2e14ef4ee4a3c2a8b8d31bc5e3187ba86db10739b4ff8ec92ff6655",
			midInitiatorL:         "13b5b6e7ac401153760e4f7fc3230c3929aa45613f11f8b875b15a9f4f8fed0e",
			midInitiatorP:         "5057b0de082b6e0f3e0b8b126598d4bdd68a3816136457228d1044dc4928913b",
			midResponderL:         "87bcdb6a4ee292789d2e9a98bb5493ed01b1fb1c4754e14700d6f8d7a5c793d3",
			midResponderP:         "448e797e388934dfe0096fd8ec512c41c52153d42ea1c509ad76435cfa097eb9",
			midSendGarbageTerm:    "160c2ec3742a2c13ccd03366f2a40130",
			midRecvGarbageTerm:    "8fa2fdc083c4596f2bd453e87257d5d2",
			outSessionID:          "5f36f31bad2ac98d9c64b917192c40583970a1ee4d1b97b4a98ae7ca370bb8f8",
			outCiphertext:         "",
			outCiphertextEndsWith: "e0ad2a576d06dfb1b994c43e31fa7eb65c0ea8fd930a241748990e78538e74d6",
		},
		{
			inIdx:                 1024,
			inPrivOurs:            "f90e080c64b05824c5a24b2501d5aeaf08af3872ee860aa80bdcd430f7b63494",
			inEllswiftOurs:        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff115173765dc202cf029ad3f15479735d57697af12b0131dd21430d5772e4ef11474d58b9",
			inEllswiftTheirs:      "12a50f3fafea7c1eeada4cf8d33777704b77361453afc83bda91eef349ae044d20126c6200547ea5a6911776c05dee2a7f1a9ba7dfbabbbd273c3ef29ef46e46",
			inInitiating:          true,
			inContents:            "5f67d15d22ca9b2804eeab0a66f7f8e3a10fa5de5809a046084348cbc5304e843ef96f59a59c7d7fdfe5946489f3ea297d941bac326225df316a25fc90f0e65b0d31a9c497e960fdbf8c482516bc8a9c1c77b7f6d0e1143810c737f76f9224e6f2c9af5186b4f7259c7e8d165b6e4fe3d38a60bdbdd4d06ecdcaaf62086070dbb68686b802d53dfd7db14b18743832605f5461ad81e2af4b7e8ff0eff0867a25b93cec7becf15c43131895fed09a83bf1ee4a87d44dd0f02a837bf5a1232e201cb882734eb9643dc2dc4d4e8b5690840766212c7ac8f38ad8a9ec47c7a9b3e022ae3eb6a32522128b518bd0d0085dd81c5",
			inMultiply:            69615,
			inAad:                 "",
			inIgnore:              true,
			midXOurs:              "8b8de966150bf872b4b695c9983df519c909811954d5d76e99ed0d5f1860247b",
			midXTheirs:            "eef379db9bd4b1aa90fc347fad33f7d53083389e22e971036f59f4e29d325ac2",
			midXShared:            "0a402d812314646ccc2565c315d1429ec1ed130ff92ff3f48d948f29c3762cf1",
			midSharedSecret:       "e25461fb0e4c162e18123ecde88342d54d449631e9b75a266fd9260c2bb2f41d",
			midInitiatorL:         "7669c7d60ab1076e6e6b5d4a79bf260ae6a527cb12ed265e5257faa6da8af60b",
			midInitiatorP:         "00793ed598c1ccc8da5b07808cd21a510d2873f764c997a0a4444e0044a742ee",
			midResponderL:         "b5a4bd6b3857be4a3462fafe9365a98d07406c10dae2078b460396e31a819490",
			midResponderP:         "b870baccca761edc9f7dc2f89bc852fbd79942fe5f03e69f63de56e1a94cda8e",
			midSendGarbageTerm:    "9b729c0d38387cd69d951390bff8238d",
			midRecvGarbageTerm:    "84ec180cc1cd3c71ac9404ea1cf1fab8",
			outSessionID:          "2d88d5c4764f1915ae9c46925aa74ca67578a37c37930853bc45a97a84c34531",
			outCiphertext:         "",
			outCiphertextEndsWith: "a9139d5e7a34bf0d95dd32e64e2cc42a38cd563186d88b5d022c2a7e73ed5ba7",
		},
	}

	for _, test := range tests {
		inInitiating := test.inInitiating

		// We need to convert the FieldVal into a ModNScalar so that we
		// can use the ScalarBaseMultNonConst.
		inPrivOurs := setHex(test.inPrivOurs)
		inPrivOursBytes := inPrivOurs.Bytes()

		var inPrivOursScalar btcec.ModNScalar
		overflow := inPrivOursScalar.SetBytes(inPrivOursBytes)
		if overflow == 1 {
			t.Fatalf("unexpected reduction")
		}

		var inPubOurs btcec.JacobianPoint
		btcec.ScalarBaseMultNonConst(&inPrivOursScalar, &inPubOurs)
		inPubOurs.ToAffine()

		midXOurs := setHex(test.midXOurs)
		if !midXOurs.Equals(&inPubOurs.X) {
			t.Fatalf("expected mid-state to match our public key")
		}

		// ellswift_decode takes in ellswift_bytes and returns a proper key.
		// 1. convert from hex to bytes
		bytesEllswiftOurs, err := hex.DecodeString(test.inEllswiftOurs)
		if err != nil {
			t.Fatalf("unexpected error decoding string")
		}

		uEllswiftOurs := bytesEllswiftOurs[:32]
		tEllswiftOurs := bytesEllswiftOurs[32:]

		var (
			uEllswiftOursFV btcec.FieldVal
			tEllswiftOursFV btcec.FieldVal
		)

		truncated := uEllswiftOursFV.SetByteSlice(uEllswiftOurs)
		if truncated {
			uEllswiftOursFV.Normalize()
		}

		truncated = tEllswiftOursFV.SetByteSlice(tEllswiftOurs)
		if truncated {
			tEllswiftOursFV.Normalize()
		}

		xEllswiftOurs, err := ellswift.XSwiftEC(
			&uEllswiftOursFV, &tEllswiftOursFV,
		)
		if err != nil {
			t.Fatalf("unexpected error during XSwiftEC")
		}

		if !midXOurs.Equals(xEllswiftOurs) {
			t.Fatalf("expected mid-state to match decoded " +
				"ellswift key")
		}

		bytesEllswiftTheirs, err := hex.DecodeString(
			test.inEllswiftTheirs,
		)
		if err != nil {
			t.Fatalf("unexpected error decoding string")
		}

		uEllswiftTheirs := bytesEllswiftTheirs[:32]
		tEllswiftTheirs := bytesEllswiftTheirs[32:]

		var (
			uEllswiftTheirsFV btcec.FieldVal
			tEllswiftTheirsFV btcec.FieldVal
		)

		truncated = uEllswiftTheirsFV.SetByteSlice(uEllswiftTheirs)
		if truncated {
			uEllswiftTheirsFV.Normalize()
		}

		truncated = tEllswiftTheirsFV.SetByteSlice(tEllswiftTheirs)
		if truncated {
			tEllswiftTheirsFV.Normalize()
		}

		xEllswiftTheirs, err := ellswift.XSwiftEC(
			&uEllswiftTheirsFV, &tEllswiftTheirsFV,
		)
		if err != nil {
			t.Fatalf("unexpected error during XSwiftEC")
		}

		midXTheirs := setHex(test.midXTheirs)
		if !midXTheirs.Equals(xEllswiftTheirs) {
			t.Fatalf("expected mid-state to match decoded " +
				"ellswift key")
		}

		privKeyOurs, _ := btcec.PrivKeyFromBytes((*inPrivOursBytes)[:])

		var bytesEllswiftTheirs64 [64]byte
		copy(bytesEllswiftTheirs64[:], bytesEllswiftTheirs)

		xShared, err := ellswift.EllswiftECDHXOnly(
			bytesEllswiftTheirs64, privKeyOurs,
		)
		if err != nil {
			t.Fatalf("unexpected error when computing shared x")
		}

		var xSharedFV btcec.FieldVal
		overflow = xSharedFV.SetBytes(&xShared)
		if overflow == 1 {
			t.Fatalf("unexpected truncation")
		}

		midXShared := setHex(test.midXShared)

		if !midXShared.Equals(&xSharedFV) {
			t.Fatalf("expected mid-state x shared")
		}

		var bytesEllswiftOurs64 [64]byte
		copy(bytesEllswiftOurs64[:], bytesEllswiftOurs)

		sharedSecret, err := ellswift.V2Ecdh(
			privKeyOurs, bytesEllswiftTheirs64,
			bytesEllswiftOurs64, inInitiating,
		)
		if err != nil {
			t.Fatalf("unexpected error when calculating " +
				"shared secret")
		}

		midShared, err := hex.DecodeString(test.midSharedSecret)
		if err != nil {
			t.Fatalf("unexpected hex decode failure")
		}

		if !bytes.Equal(midShared, sharedSecret[:]) {
			t.Fatalf("expected mid shared secret")
		}

		p := NewPeer()

		buf := bytes.NewBuffer(nil)
		p.UseReadWriter(buf)

		err = p.createV2Ciphers(midShared, inInitiating, mainNet)
		if err != nil {
			t.Fatalf("error initiating v2 transport")
		}

		midInitiatorL, err := hex.DecodeString(test.midInitiatorL)
		if err != nil {
			t.Fatalf("unexpected error decoding midInitiatorL")
		}

		if !bytes.Equal(midInitiatorL, p.initiatorL) {
			t.Fatalf("expected mid-state initiatorL to " +
				"match computed value")
		}

		midInitiatorP, err := hex.DecodeString(test.midInitiatorP)
		if err != nil {
			t.Fatalf("unexpected error decoding midInitiatorP")
		}

		if !bytes.Equal(midInitiatorP, p.initiatorP) {
			t.Fatalf("expected mid-state initiatorP to " +
				"match computed value")
		}

		midResponderL, err := hex.DecodeString(test.midResponderL)
		if err != nil {
			t.Fatalf("unexpected error decoding midResponderL")
		}

		if !bytes.Equal(midResponderL, p.responderL) {
			t.Fatalf("expected mid-state responderL to " +
				"match computed value")
		}

		midResponderP, err := hex.DecodeString(test.midResponderP)
		if err != nil {
			t.Fatalf("unexpected error decoding midResponderP")
		}

		if !bytes.Equal(midResponderP, p.responderP) {
			t.Fatalf("expected mid-state responderP to " +
				"match computed value")
		}

		midSendGarbageTerm, err := hex.DecodeString(
			test.midSendGarbageTerm,
		)
		if err != nil {
			t.Fatalf("unexpected error decoding midSendGarbageTerm")
		}

		if !bytes.Equal(midSendGarbageTerm, p.sendGarbageTerm[:]) {
			t.Fatalf("expected mid-state sendGarbageTerm " +
				"to match computed value")
		}

		midRecvGarbageTerm, err := hex.DecodeString(
			test.midRecvGarbageTerm,
		)
		if err != nil {
			t.Fatalf("unexpected error decoding midRecvGarbageTerm")
		}

		if !bytes.Equal(midRecvGarbageTerm, p.recvGarbageTerm) {
			t.Fatalf("expected mid-state recvGarbageTerm to " +
				"match computed value")
		}

		outSessionID, err := hex.DecodeString(test.outSessionID)
		if err != nil {
			t.Fatalf("unexpected error decoding outSessionID")
		}

		if !bytes.Equal(outSessionID, p.sessionID) {
			t.Fatalf("expected sessionID to match computed value")
		}

		for i := 0; i < test.inIdx; i++ {
			_, _, err = p.V2EncPacket([]byte{}, []byte{}, false)
			if err != nil {
				t.Fatalf("unexpected error while encrypting packet")
			}
		}

		initialContents, err := hex.DecodeString(test.inContents)
		if err != nil {
			t.Fatalf("unexpected error decoding contents")
		}

		aad, err := hex.DecodeString(test.inAad)
		if err != nil {
			t.Fatalf("unexpected error decoding aad")
		}

		var contents []byte

		copy(contents, initialContents)

		for i := 0; i < test.inMultiply; i++ {
			contents = append(contents, initialContents...)
		}

		ciphertext, _, err := p.V2EncPacket(contents, aad, test.inIgnore)
		if err != nil {
			t.Fatalf("unexpected error when encrypting packet: %v", err)
		}

		if len(test.outCiphertext) != 0 {
			outCiphertextBytes, err := hex.DecodeString(test.outCiphertext)
			if err != nil {
				t.Fatalf("unexpected error decoding outCiphertext: %v", err)
			}

			if !bytes.Equal(outCiphertextBytes, ciphertext) {
				t.Fatalf("ciphertext mismatch")
			}
		}

		if len(test.outCiphertextEndsWith) != 0 {
			ciphertextHex := hex.EncodeToString(ciphertext)
			if !strings.HasSuffix(ciphertextHex, test.outCiphertextEndsWith) {
				t.Fatalf("suffix mismatch")
			}
		}
	}
}

type recordingAdmission struct {
	acquired, released int
}

func (a *recordingAdmission) Acquire() (func(), error) {
	a.acquired++
	return func() { a.released++ }, nil
}

// TestDeriveV2CiphersReleasesLease pins that the responder lease is returned on success and when the derivation
// panics; the peer goroutine recovers panics, so a leaked lease would never come back.
func TestDeriveV2CiphersReleasesLease(t *testing.T) {
	t.Parallel()

	_, ellswiftTheirs, err := ellswift.EllswiftCreate()
	require.NoError(t, err)

	newResponder := func(t *testing.T) (*Peer, *recordingAdmission) {
		t.Helper()

		admission := &recordingAdmission{}
		p := NewPeerWithOptions(WithResponderHandshakeAdmission(admission))
		var err error
		p.privkeyOurs, p.ellswiftOurs, err = ellswift.EllswiftCreate()
		require.NoError(t, err)
		return p, admission
	}

	t.Run("released on success", func(t *testing.T) {
		p, admission := newResponder(t)
		require.NoError(t, p.deriveV2Ciphers(ellswiftTheirs, false, PearlNet(1)))
		assert.Equal(t, 1, admission.acquired)
		assert.Equal(t, 1, admission.released)
	})

	t.Run("released when the derivation panics", func(t *testing.T) {
		p, admission := newResponder(t)
		p.privkeyOurs = nil
		require.Panics(t, func() { _ = p.deriveV2Ciphers(ellswiftTheirs, false, PearlNet(1)) })
		assert.Equal(t, 1, admission.acquired)
		assert.Equal(t, 1, admission.released)
	})

	t.Run("initiator takes no lease", func(t *testing.T) {
		p, admission := newResponder(t)
		require.NoError(t, p.deriveV2Ciphers(ellswiftTheirs, true, PearlNet(1)))
		assert.Zero(t, admission.acquired)
	})
}
