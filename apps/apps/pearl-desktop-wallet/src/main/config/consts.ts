// DNS seeders used by the oyster daemon for automatic peer discovery.
// These match the DNSSeeds in node/chaincfg/params.go and are queried by the
// node itself at startup — no --addpeer flag is needed.
export const MAINNET_DNS_SEEDERS = [
  'seeder1.pearlresearch.ai',
  'seeder2.pearlresearch.ai',
  'seeder3.pearlresearch.ai',
];
export const TESTNET_DNS_SEEDERS = [
  'seeder1.testnet.pearlresearch.ai',
  'seeder2.testnet.pearlresearch.ai',
  'seeder3.testnet.pearlresearch.ai',
];

// Legacy hardcoded peer hosts from older wallet versions. Kept only so stale
// peer-settings.json entries (saved as custom --addpeer targets) are detected
// and ignored, falling back to DNS-seeder-based discovery.
export const LEGACY_MAINNET_PEER_ADDRESSES = [
  'wallet-node0.pearlresearch.ai',
  'wallet-node1.pearlresearch.ai',
  'wallet-node2.pearlresearch.ai',
  'wallet-node3.pearlresearch.ai',
  'wallet-node4.pearlresearch.ai',
];
export const LEGACY_TESTNET_PEER_ADDRESSES = [
  'node1.testnet.pearlresearch.ai',
  'node2.testnet.pearlresearch.ai',
  'node3.testnet.pearlresearch.ai',
];

export const UPDATE_REPO_OWNER = 'pearl-research-labs';
export const UPDATE_REPO_NAME = 'pearl';
export const UPDATE_RELEASE_TAG_PREFIX = 'pearl-wallet-v';
