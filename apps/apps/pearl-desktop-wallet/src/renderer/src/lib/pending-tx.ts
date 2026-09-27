// The daemon reports this when it announced a transaction but no peer asked for it. On a first send nothing left this
// machine; on a rebroadcast it is ambiguous, because peers that already hold the transaction stay silent.
export function isNotRelayedError(message: string): boolean {
  return message.includes('not relayed');
}
