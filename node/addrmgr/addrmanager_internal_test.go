package addrmgr

import (
	"encoding/json"
	"errors"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/wire"
)

// randAddr generates a *wire.NetAddressV2 backed by a random IPv4/IPv6
// address.  Some of the returned addresses may not be routable.
func randAddr(t *testing.T) *wire.NetAddressV2 {
	t.Helper()

	ipv4 := rand.Intn(2) == 0
	var ip net.IP
	if ipv4 {
		var b [4]byte
		if _, err := rand.Read(b[:]); err != nil {
			t.Fatal(err)
		}
		ip = b[:]
	} else {
		var b [16]byte
		if _, err := rand.Read(b[:]); err != nil {
			t.Fatal(err)
		}
		ip = b[:]
	}

	services := wire.ServiceFlag(rand.Uint64())
	port := uint16(rand.Uint32())

	return wire.NetAddressV2FromBytes(
		time.Now(), services, ip, port,
	)
}

// routableRandAddr generates a *wire.NetAddressV2 backed by a random IPv4/IPv6
// address that is always routable.
func routableRandAddr(t *testing.T) *wire.NetAddressV2 {
	t.Helper()

	var addr *wire.NetAddressV2

	// If the address is not routable, try again.
	routable := false
	for !routable {
		addr = randAddr(t)
		routable = IsRoutable(addr)
	}

	return addr
}

// assertAddr ensures that the two addresses match. The timestamp is not
// checked as it does not affect uniquely identifying a specific address.
func assertAddr(t *testing.T, got, expected *wire.NetAddressV2) {
	if got.Services != expected.Services {
		t.Fatalf("expected address services %v, got %v",
			expected.Services, got.Services)
	}
	gotAddr := got.Addr.String()
	expectedAddr := expected.Addr.String()
	if gotAddr != expectedAddr {
		t.Fatalf("expected address IP %v, got %v", expectedAddr,
			gotAddr)
	}
	if got.Port != expected.Port {
		t.Fatalf("expected address port %d, got %d", expected.Port,
			got.Port)
	}
}

// assertAddrs ensures that the manager's address cache matches the given
// expected addresses.
func assertAddrs(t *testing.T, addrMgr *AddrManager,
	expectedAddrs map[string]*wire.NetAddressV2) {

	t.Helper()

	addrs := addrMgr.getAddresses()

	if len(addrs) != len(expectedAddrs) {
		t.Fatalf("expected to find %d addresses, found %d",
			len(expectedAddrs), len(addrs))
	}

	for _, addr := range addrs {
		addrStr := NetAddressKey(addr)
		expectedAddr, ok := expectedAddrs[addrStr]
		if !ok {
			t.Fatalf("expected to find address %v", addrStr)
		}

		assertAddr(t, addr, expectedAddr)
	}
}

// TestAddrManagerSerialization ensures that we can properly serialize and
// deserialize the manager's current address cache.
func TestAddrManagerSerialization(t *testing.T) {
	t.Parallel()

	// We'll start by creating our address manager backed by a temporary
	// directory.
	tempDir := t.TempDir()

	addrMgr := New(tempDir, nil)

	// We'll be adding 5 random addresses to the manager.
	const numAddrs = 5

	expectedAddrs := make(map[string]*wire.NetAddressV2, numAddrs)
	for i := 0; i < numAddrs; i++ {
		addr := routableRandAddr(t)
		expectedAddrs[NetAddressKey(addr)] = addr
		addrMgr.AddAddress(addr, routableRandAddr(t))
	}

	// Now that the addresses have been added, we should be able to retrieve
	// them.
	assertAddrs(t, addrMgr, expectedAddrs)

	// Then, we'll persist these addresses to disk and restart the address
	// manager.
	addrMgr.savePeers()
	addrMgr = New(tempDir, nil)

	// Finally, we'll read all of the addresses from disk and ensure they
	// match as expected.
	addrMgr.loadPeers()
	assertAddrs(t, addrMgr, expectedAddrs)
}

// TestAddrManagerV1ToV2 ensures that we can properly upgrade the serialized
// version of the address manager from v1 to v2.
func TestAddrManagerV1ToV2(t *testing.T) {
	t.Parallel()

	// We'll start by creating our address manager backed by a temporary
	// directory.
	tempDir := t.TempDir()

	addrMgr := New(tempDir, nil)

	// As we're interested in testing the upgrade path from v1 to v2, we'll
	// override the manager's current version.
	addrMgr.version = 1

	// We'll be adding 5 random addresses to the manager. Since this is v1,
	// each addresses' services will not be stored.
	const numAddrs = 5

	expectedAddrs := make(map[string]*wire.NetAddressV2, numAddrs)
	for i := 0; i < numAddrs; i++ {
		addr := routableRandAddr(t)
		expectedAddrs[NetAddressKey(addr)] = addr
		addrMgr.AddAddress(addr, routableRandAddr(t))
	}

	// Then, we'll persist these addresses to disk and restart the address
	// manager - overriding its version back to v1.
	addrMgr.savePeers()
	addrMgr = New(tempDir, nil)
	addrMgr.version = 1

	// When we read all of the addresses back from disk, we should expect to
	// find all of them, but their services will be set to a default of
	// SFNodeNetwork since they were not previously stored. After ensuring
	// that this default is set, we'll override each addresses' services
	// with the original value from when they were created.
	addrMgr.loadPeers()
	addrs := addrMgr.getAddresses()
	if len(addrs) != len(expectedAddrs) {
		t.Fatalf("expected to find %d addresses, found %d",
			len(expectedAddrs), len(addrs))
	}
	for _, addr := range addrs {
		addrStr := NetAddressKey(addr)
		expectedAddr, ok := expectedAddrs[addrStr]
		if !ok {
			t.Fatalf("expected to find address %v", addrStr)
		}

		if addr.Services != wire.SFNodeNetwork {
			t.Fatalf("expected address services to be %v, got %v",
				wire.SFNodeNetwork, addr.Services)
		}

		addrMgr.SetServices(addr, expectedAddr.Services)
	}

	// We'll also bump up the manager's version to v2, which should signal
	// that it should include the address services when persisting its
	// state.
	addrMgr.version = 2
	addrMgr.savePeers()

	// Finally, we'll recreate the manager and ensure that the services were
	// persisted correctly.
	addrMgr = New(tempDir, nil)
	addrMgr.loadPeers()
	assertAddrs(t, addrMgr, expectedAddrs)
}

// TestLoadPeersFailedResetsCounters ensures that when peers.json is
// valid JSON but internally inconsistent (an address listed in
// Addresses that appears in no bucket and is not tried), the sanity
// check in deserializePeers fails after it has already counted the
// bucketed addresses — and loadPeers' "start fresh" reset leaves the
// manager truly empty. Before the fix, reset() rebuilt the buckets
// but left nNew/nTried stale, so NumAddresses reported phantom
// addresses and GetAddress spun forever on empty buckets while
// holding the manager lock.
func TestLoadPeersFailedResetsCounters(t *testing.T) {
	dir := t.TempDir()

	sam := serializedAddrManager{Version: serialisationVersion}
	sam.Addresses = []*serializedKnownAddress{
		{Addr: "1.2.3.4:44108", Src: "1.2.3.4:44108", Services: wire.SFNodeNetwork, SrcServices: wire.SFNodeNetwork},
		{Addr: "5.6.7.8:44108", Src: "5.6.7.8:44108", Services: wire.SFNodeNetwork, SrcServices: wire.SFNodeNetwork},
	}
	// Only the first address is bucketed; the second trips the
	// "no references" sanity check, after nNew has been counted.
	sam.NewBuckets[0] = []string{"1.2.3.4:44108"}

	raw, err := json.Marshal(sam)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "peers.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}

	am := New(dir, func(string) ([]net.IP, error) {
		return nil, errors.New("not implemented")
	})
	am.loadPeers()

	if got := am.NumAddresses(); got != 0 {
		t.Fatalf("NumAddresses = %d after failed load + reset, want 0", got)
	}

	done := make(chan *KnownAddress, 1)
	go func() { done <- am.GetAddress() }()
	select {
	case ka := <-done:
		if ka != nil {
			t.Fatalf("GetAddress = %v after failed load + reset, want nil", ka)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("GetAddress did not return within 5s (spins on empty buckets)")
	}
}
