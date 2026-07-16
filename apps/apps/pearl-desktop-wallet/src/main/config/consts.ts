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
