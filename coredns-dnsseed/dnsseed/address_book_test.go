package dnsseed

import (
	"fmt"
	"net/netip"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testDefaultPort = "44108"

func mustAddr(s string) netip.AddrPort {
	return netip.MustParseAddrPort(s)
}

func TestAddressBook_AddAndCount(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	assert.Equal(t, 0, ab.count())

	ab.add(mustAddr("127.0.0.1:44108"))
	assert.Equal(t, 1, ab.count())

	ab.add(mustAddr("127.0.0.2:44108"))
	assert.Equal(t, 2, ab.count())

	ab.add(mustAddr("127.0.0.1:44108"))
	assert.Equal(t, 2, ab.count())
}

func TestAddressBook_AddSkipsNonDefaultPort(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	ab.add(mustAddr("127.0.0.1:9999"))
	assert.Equal(t, 0, ab.count())
	assert.False(t, ab.isKnown(mustAddr("127.0.0.1:9999")))
}

func TestAddressBook_AddRespectsCap(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	for i := range maxAddressBookSize {
		ab.add(mustAddr(fmt.Sprintf("10.0.%d.%d:44108", i/256, i%256)))
	}
	require.Equal(t, maxAddressBookSize, ab.count())

	overflow := mustAddr("192.168.1.1:44108")
	ab.add(overflow)
	assert.Equal(t, maxAddressBookSize, ab.count())
	assert.False(t, ab.isKnown(overflow))
}

func TestAddressBook_IsKnown(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add(mustAddr("127.0.0.1:44108"))

	assert.True(t, ab.isKnown(mustAddr("127.0.0.1:44108")))
	assert.False(t, ab.isKnown(mustAddr("10.0.0.1:44108")))
}

// TestAddressBook_MarkFailedUnverified verifies that a gossiped address that
// was never booked enters cooldown on its first failure.
func TestAddressBook_MarkFailedUnverified(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	addr := mustAddr("10.0.0.1:44108")

	ab.markFailed(addr)
	assert.True(t, ab.isCoolingDown(addr))
	assert.True(t, ab.isKnown(addr),
		"cooling-down addresses are known so gossip does not re-dial them")
	assert.Equal(t, 0, ab.count())
}

// TestAddressBook_TwoStrikes verifies that a booked (verified) peer keeps
// being served after one failure and only enters cooldown at the second.
func TestAddressBook_TwoStrikes(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	addr := mustAddr("127.0.0.1:44108")
	ab.add(addr)

	ab.markFailed(addr)
	assert.Equal(t, 1, ab.count(), "one failure must not unserve a verified peer")
	assert.False(t, ab.isCoolingDown(addr))
	assert.Len(t, ab.shuffledAddresses(false), 1)

	ab.markFailed(addr)
	assert.Equal(t, 0, ab.count(), "second consecutive failure must unserve the peer")
	assert.True(t, ab.isCoolingDown(addr))
	assert.Empty(t, ab.shuffledAddresses(false))
}

func TestAddressBook_AddResetsFailures(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	addr := mustAddr("127.0.0.1:44108")
	ab.add(addr)

	ab.markFailed(addr)
	ab.add(addr)

	ab.markFailed(addr)
	assert.Equal(t, 1, ab.count(),
		"failures separated by a successful handshake must not accumulate")
}

// TestAddressBook_CooldownExpires verifies that expired cooldown entries are
// treated as absent even before the prune sweep runs.
func TestAddressBook_CooldownExpires(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	addr := mustAddr("10.0.0.1:44108")

	ab.markFailed(addr)
	expireCooldown(ab, addr)

	assert.False(t, ab.isCoolingDown(addr))
	assert.False(t, ab.isKnown(addr),
		"an expired entry must be re-dialable by gossip")
}

func TestAddressBook_PruneCooldown(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	expired := mustAddr("10.0.0.1:44108")
	active := mustAddr("10.0.0.2:44108")

	ab.markFailed(expired)
	ab.markFailed(active)
	expireCooldown(ab, expired)

	ab.pruneCooldown()

	ab.mu.RLock()
	defer ab.mu.RUnlock()
	assert.NotContains(t, ab.failedAt, expired)
	assert.Contains(t, ab.failedAt, active)
}

func TestAddressBook_ShuffledAddresses(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add(mustAddr("127.0.0.1:44108"))
	ab.add(mustAddr("127.0.0.2:44108"))
	ab.add(mustAddr("127.0.0.3:44108"))

	ips := ab.shuffledAddresses(false)
	require.Len(t, ips, 3)
	assert.NotEqual(t, ips[0].String(), ips[1].String())
	assert.NotEqual(t, ips[0].String(), ips[2].String())
	assert.NotEqual(t, ips[1].String(), ips[2].String())
}

func TestAddressBook_ShuffledAddressesV6(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add(mustAddr("127.0.0.1:44108"))
	ab.add(mustAddr("[::1]:44108"))

	v4 := ab.shuffledAddresses(false)
	v6 := ab.shuffledAddresses(true)

	assert.Len(t, v4, 1)
	assert.Len(t, v6, 1)
}
