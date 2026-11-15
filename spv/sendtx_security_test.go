package neutrino

import (
	"net"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pearl-research-labs/pearl/node/chaincfg"
	"github.com/pearl-research-labs/pearl/node/chaincfg/chainhash"
	"github.com/pearl-research-labs/pearl/node/peer"
	"github.com/pearl-research-labs/pearl/node/wire"
	"github.com/pearl-research-labs/pearl/spv/pushtx"
	"github.com/stretchr/testify/require"
)

type broadcastPeerBehavior struct {
	requestTypes    []wire.InvType
	repeatRequest   int
	rejectOnInv     bool
	rejectCommand   string
	wrongRejectHash bool
	repeatReject    int
}

type broadcastTestPeer struct {
	server     *ServerPeer
	announced  atomic.Int32
	txSent     atomic.Int32
	txReceived atomic.Int32
}

// newBroadcastTestPeer uses the real peer encoders and readers so that a rejection must arrive over the wire through
// the normal SPV subscription.
func newBroadcastTestPeer(t *testing.T, service *ChainService, behavior broadcastPeerBehavior) *broadcastTestPeer {
	t.Helper()
	result := &broadcastTestPeer{server: NewServerPeer(service, false)}
	listener, err := net.ListenTCP("tcp", &net.TCPAddr{IP: net.ParseIP("127.0.0.1")})
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	require.NoError(t, listener.SetDeadline(time.Now().Add(5*time.Second)))

	local, err := peer.NewOutboundPeer(&peer.Config{
		ChainParams:      &chaincfg.RegressionNetParams,
		AllowSelfConns:   true,
		UserAgentName:    "broadcast-local",
		UserAgentVersion: "1",
		Listeners: peer.MessageListeners{
			OnRead: result.server.OnRead,
			OnWrite: func(_ *peer.Peer, _ int, msg wire.Message, err error) {
				if _, ok := msg.(*wire.MsgTx); ok && err == nil {
					result.txSent.Add(1)
				}
			},
		},
	}, listener.Addr().String())
	require.NoError(t, err)
	result.server.Peer = local

	reject := func(remote *peer.Peer, hash chainhash.Hash) {
		if behavior.wrongRejectHash {
			hash[0] ^= 1
		}
		for i := 0; i < max(1, behavior.repeatReject); i++ {
			message := wire.NewMsgReject(behavior.rejectCommand, wire.RejectInvalid, "test-invalid")
			message.Hash = hash
			remote.QueueMessage(message, nil)
		}
	}
	remote := peer.NewInboundPeer(&peer.Config{
		ChainParams:      &chaincfg.RegressionNetParams,
		AllowSelfConns:   true,
		UserAgentName:    "broadcast-remote",
		UserAgentVersion: "1",
		Listeners: peer.MessageListeners{
			OnInv: func(remote *peer.Peer, inv *wire.MsgInv) {
				for _, vector := range inv.InvList {
					result.announced.Add(1)
					if behavior.rejectOnInv {
						reject(remote, vector.Hash)
						continue
					}
					request := wire.NewMsgGetData()
					for _, inventoryType := range behavior.requestTypes {
						request.InvList = append(request.InvList,
							wire.NewInvVect(inventoryType, &vector.Hash))
					}
					for i := 0; i < max(1, behavior.repeatRequest); i++ {
						remote.QueueMessage(request, nil)
					}
				}
			},
			OnTx: func(remote *peer.Peer, tx *wire.MsgTx) {
				result.txReceived.Add(1)
				if behavior.rejectCommand != "" {
					reject(remote, tx.TxHash())
				}
			},
		},
	})
	t.Cleanup(func() {
		local.Disconnect()
		remote.Disconnect()
	})

	localConn, err := net.DialTimeout("tcp", listener.Addr().String(), 5*time.Second)
	require.NoError(t, err)
	t.Cleanup(func() { _ = localConn.Close() })
	remoteConn, err := listener.Accept()
	require.NoError(t, err)
	t.Cleanup(func() { _ = remoteConn.Close() })
	require.NoError(t, listener.Close())
	local.AssociateConnection(localConn)
	remote.AssociateConnection(remoteConn)
	require.Eventually(t, func() bool {
		return local.ID() != 0 && local.VerAckReceived() && remote.VerAckReceived()
	}, 5*time.Second, 5*time.Millisecond, "peer handshake did not complete")
	return result
}

func broadcastWithTestPeers(t *testing.T, behaviors []broadcastPeerBehavior) (error, []*broadcastTestPeer) {
	t.Helper()
	service := &ChainService{
		query:            make(chan interface{}),
		quit:             make(chan struct{}),
		broadcastTimeout: time.Second,
	}
	peers := make([]*broadcastTestPeer, 0, len(behaviors))
	serverPeers := make([]*ServerPeer, 0, len(behaviors))
	for _, behavior := range behaviors {
		p := newBroadcastTestPeer(t, service, behavior)
		peers = append(peers, p)
		serverPeers = append(serverPeers, p.server)
	}
	queryDone := make(chan struct{})
	go func() {
		defer close(queryDone)
		for {
			select {
			case query := <-service.query:
				switch request := query.(type) {
				case getPeersMsg:
					select {
					case request.reply <- serverPeers:
					case <-service.quit:
						return
					}
				case getConnCountMsg:
					select {
					case request.reply <- int32(len(serverPeers)):
					case <-service.quit:
						return
					}
				}
			case <-service.quit:
				return
			}
		}
	}()
	t.Cleanup(func() {
		close(service.quit)
		select {
		case <-queryDone:
		case <-time.After(5 * time.Second):
			t.Error("peer query handler did not stop")
		}
	})

	tx := wire.NewMsgTx(wire.TxVersion)
	tx.AddTxIn(&wire.TxIn{})
	tx.AddTxOut(&wire.TxOut{Value: 1, PkScript: []byte{0x51}})
	result := make(chan error, 1)
	go func() {
		result <- service.sendTransaction(tx, Timeout(time.Second), RejectTimeout(250*time.Millisecond))
	}()
	select {
	case err := <-result:
		for i, p := range peers {
			require.Positive(t, p.announced.Load(), "peer %d did not receive inv", i)
		}
		return err, peers
	case <-time.After(5 * time.Second):
		t.Fatal("transaction broadcast did not finish")
		return nil, peers
	}
}

func TestSendTransactionRejectPeerSets(t *testing.T) {
	fetch := broadcastPeerBehavior{requestTypes: []wire.InvType{wire.InvTypeTx}}
	rejectOnly := broadcastPeerBehavior{rejectOnInv: true, rejectCommand: wire.CmdTx}
	fetchReject := broadcastPeerBehavior{
		requestTypes: []wire.InvType{wire.InvTypeTx}, rejectCommand: wire.CmdTx,
	}
	duplicateFetch := broadcastPeerBehavior{
		requestTypes:  []wire.InvType{wire.InvTypeTx, wire.InvTypeTx, wire.InvTypeTx},
		repeatRequest: 3,
	}
	duplicateReject := fetchReject
	duplicateReject.repeatReject = 5

	tests := []struct {
		name        string
		behaviors   []broadcastPeerBehavior
		wantInvalid bool
	}{
		{"five_requesters_three_reject_only", []broadcastPeerBehavior{
			fetch, fetch, fetch, fetch, fetch, rejectOnly, rejectOnly, rejectOnly,
		}, false},
		{"equal_count_disjoint_peer_sets", []broadcastPeerBehavior{
			fetch, fetch, fetch, rejectOnly, rejectOnly, rejectOnly,
		}, false},
		{"all_requesters_reject", []broadcastPeerBehavior{fetchReject, fetchReject}, true},
		{"requester_rejections_above_threshold", []broadcastPeerBehavior{
			fetchReject, fetchReject, fetchReject, fetchReject, fetchReject, fetch, fetch, fetch,
		}, true},
		{"requester_rejections_below_threshold", []broadcastPeerBehavior{
			fetchReject, fetchReject, fetchReject, fetchReject, fetch, fetch, fetch, fetch,
		}, false},
		{"duplicate_requests_count_one_peer", []broadcastPeerBehavior{
			fetchReject, fetchReject, duplicateFetch,
		}, true},
		{"duplicate_rejections_count_one_peer", []broadcastPeerBehavior{
			duplicateReject, fetch, fetch,
		}, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err, peers := broadcastWithTestPeers(t, test.behaviors)
			if test.wantInvalid {
				require.Truef(t, pushtx.IsBroadcastError(err, pushtx.Invalid), "got %v", err)
			} else {
				require.NoError(t, err)
			}
			for i, p := range peers {
				if test.behaviors[i].rejectOnInv {
					require.Zero(t, p.txSent.Load(),
						"reject-only peer %d received transaction bytes", i)
				} else {
					require.EqualValues(t, 1, p.txReceived.Load(),
						"requester %d should receive one transaction", i)
				}
			}
		})
	}
}

func TestSendTransactionRejectIdentity(t *testing.T) {
	for _, test := range []struct {
		name      string
		command   string
		wrongHash bool
	}{
		{"wrong_command", wire.CmdBlock, false},
		{"wrong_hash", wire.CmdTx, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			err, peers := broadcastWithTestPeers(t, []broadcastPeerBehavior{{
				requestTypes:    []wire.InvType{wire.InvTypeTx},
				rejectCommand:   test.command,
				wrongRejectHash: test.wrongHash,
			}})
			require.NoError(t, err)
			require.EqualValues(t, 1, peers[0].txReceived.Load())
		})
	}
}

func TestSendTransactionGetDataInventoryTypes(t *testing.T) {
	for _, inventoryType := range []wire.InvType{
		wire.InvTypeTx, wire.InvTypeWitnessTx,
		wire.InvTypeError, wire.InvTypeBlock, wire.InvTypeWitnessBlock,
		wire.InvTypeFilteredBlock, wire.InvTypeFilteredWitnessBlock,
	} {
		t.Run(inventoryType.String(), func(t *testing.T) {
			err, peers := broadcastWithTestPeers(t, []broadcastPeerBehavior{{
				requestTypes: []wire.InvType{inventoryType},
			}})
			if inventoryType == wire.InvTypeTx || inventoryType == wire.InvTypeWitnessTx {
				require.NoError(t, err)
				require.EqualValues(t, 1, peers[0].txReceived.Load())
				return
			}
			require.Truef(t, pushtx.IsBroadcastError(err, pushtx.NotRelayed), "got %v", err)
			require.Zero(t, peers[0].txSent.Load(), "non-transaction getdata sent transaction bytes")
			require.Zero(t, peers[0].txReceived.Load(), "non-transaction getdata received a transaction")
		})
	}
}
