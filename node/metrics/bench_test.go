package metrics

import "testing"

// RecordWireMessage runs on every P2P frame in both directions, so it is the
// only recording helper whose cost is worth tracking.  newPeerConfig in
// node/server.go skips installing the hooks entirely when no metrics endpoint is
// configured, so these numbers only apply when metrics are in use.
func BenchmarkRecordWireMessage(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		RecordWireMessage(true, "inv", 128)
	}
}

// Every peer's read and write handler shares one CounterVec, so measure the
// contention a full peer set would produce.
func BenchmarkRecordWireMessageParallel(b *testing.B) {
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			RecordWireMessage(true, "inv", 128)
		}
	})
}

// Spread across commands so the label lookup misses the same map slot every
// time, approximating real mixed traffic.
func BenchmarkRecordWireMessageMixedCommands(b *testing.B) {
	commands := []string{"inv", "tx", "block", "ping", "pong", "getdata", "headers", "addr"}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		RecordWireMessage(i%2 == 0, commands[i%len(commands)], 128)
	}
}
