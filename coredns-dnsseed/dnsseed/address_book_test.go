package dnsseed

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testDefaultPort = "44108"

func TestAddressBook_AddAndCount(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	assert.Equal(t, 0, ab.count())

	ab.add("127.0.0.1:44108")
	assert.Equal(t, 1, ab.count())

	ab.add("127.0.0.2:44108")
	assert.Equal(t, 2, ab.count())

	// Duplicate should overwrite, not increase count.
	ab.add("127.0.0.1:44108")
	assert.Equal(t, 2, ab.count())
}

func TestAddressBook_AddSkipsNonDefaultPort(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	ab.add("127.0.0.1:9999")
	assert.Equal(t, 0, ab.count())
	assert.False(t, ab.isKnown("127.0.0.1:9999"))
}

func TestAddressBook_AddRespectsCap(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	for i := range maxAddressBookSize {
		ab.add(peerKey(fmt.Sprintf("10.0.%d.%d:44108", i/256, i%256)))
	}
	require.Equal(t, maxAddressBookSize, ab.count())

	ab.add("192.168.1.1:44108")
	assert.Equal(t, maxAddressBookSize, ab.count())
	assert.False(t, ab.isKnown("192.168.1.1:44108"))
}

func TestAddressBook_IsKnown(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add("127.0.0.1:44108")

	assert.True(t, ab.isKnown("127.0.0.1:44108"))
	assert.False(t, ab.isKnown("10.0.0.1:44108"))
}

// TestAddressBook_MarkFailedUnverified verifies that a gossiped address that
// was never booked enters cooldown on its first failure.
func TestAddressBook_MarkFailedUnverified(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	ab.markFailed("10.0.0.1:44108")
	assert.True(t, ab.isCoolingDown("10.0.0.1:44108"))
	assert.True(t, ab.isKnown("10.0.0.1:44108"),
		"cooling-down addresses are known so gossip does not re-dial them")
	assert.Equal(t, 0, ab.count())
}

// TestAddressBook_TwoStrikes verifies that a booked (verified) peer keeps
// being served after one failure and only enters cooldown at the second.
func TestAddressBook_TwoStrikes(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add("127.0.0.1:44108")

	ab.markFailed("127.0.0.1:44108")
	assert.Equal(t, 1, ab.count(), "one failure must not unserve a verified peer")
	assert.False(t, ab.isCoolingDown("127.0.0.1:44108"))
	assert.Len(t, ab.shuffleAddressList(10, false), 1)

	ab.markFailed("127.0.0.1:44108")
	assert.Equal(t, 0, ab.count(), "second consecutive failure must unserve the peer")
	assert.True(t, ab.isCoolingDown("127.0.0.1:44108"))
	assert.Empty(t, ab.shuffleAddressList(10, false))
}

// TestAddressBook_TouchResetsFailures verifies that a successful handshake
// resets the failure counter, so failures must be consecutive to unserve.
func TestAddressBook_TouchResetsFailures(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add("127.0.0.1:44108")

	ab.markFailed("127.0.0.1:44108")
	assert.True(t, ab.touch("127.0.0.1:44108"))

	ab.markFailed("127.0.0.1:44108")
	assert.Equal(t, 1, ab.count(),
		"failures separated by a successful handshake must not accumulate")
}

func TestAddressBook_TouchUnknownPeer(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	assert.False(t, ab.touch("10.0.0.1:44108"))
	assert.Equal(t, 0, ab.count())
}

// TestAddressBook_CooldownExpires verifies that expired cooldown entries are
// treated as absent even before the prune sweep runs.
func TestAddressBook_CooldownExpires(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	ab.markFailed("10.0.0.1:44108")
	expireCooldown(ab, "10.0.0.1:44108")

	assert.False(t, ab.isCoolingDown("10.0.0.1:44108"))
	assert.False(t, ab.isKnown("10.0.0.1:44108"),
		"an expired entry must be re-dialable by gossip")
}

func TestAddressBook_PruneCooldown(t *testing.T) {
	ab := newAddressBook(testDefaultPort)

	ab.markFailed("10.0.0.1:44108")
	ab.markFailed("10.0.0.2:44108")
	expireCooldown(ab, "10.0.0.1:44108")

	ab.pruneCooldown()

	ab.mu.RLock()
	defer ab.mu.RUnlock()
	assert.NotContains(t, ab.failedAt, peerKey("10.0.0.1:44108"))
	assert.Contains(t, ab.failedAt, peerKey("10.0.0.2:44108"))
}

func TestAddressBook_ShuffleAddressList(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add("127.0.0.1:44108")
	ab.add("127.0.0.2:44108")
	ab.add("127.0.0.3:44108")

	ips := ab.shuffleAddressList(10, false)
	assert.Len(t, ips, 3)

	// Capped sampling returns distinct entries.
	for range 20 {
		ips = ab.shuffleAddressList(2, false)
		assert.Len(t, ips, 2)
		assert.NotEqual(t, ips[0].String(), ips[1].String())
	}
}

func TestAddressBook_ShuffleAddressListV6(t *testing.T) {
	ab := newAddressBook(testDefaultPort)
	ab.add("127.0.0.1:44108")
	ab.add("[::1]:44108")

	v4 := ab.shuffleAddressList(10, false)
	v6 := ab.shuffleAddressList(10, true)

	assert.Len(t, v4, 1)
	assert.Len(t, v6, 1)
}
