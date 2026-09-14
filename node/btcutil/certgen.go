// Copyright (c) 2025-2026 The Pearl Research Labs
// Use of this source code is governed by an ISC
// license that can be found in the LICENSE file.

package btcutil

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	_ "crypto/sha512" // Needed for RegisterHash in init
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"
)

// NewTLSCertPair returns a new PEM-encoded x.509 certificate pair
// based on a P-256 ECDSA private key.  The machine's local interface
// addresses and all variants of IPv4 and IPv6 localhost are included as
// valid IP addresses.
func NewTLSCertPair(organization string, validUntil time.Time, extraHosts []string) (cert, key []byte, err error) {
	now := time.Now()
	if validUntil.Before(now) {
		return nil, nil, errors.New("validUntil would create an already-expired certificate")
	}

	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, err
	}

	// end of ASN.1 time
	endOfTime := time.Date(2049, 12, 31, 23, 59, 59, 0, time.UTC)
	if validUntil.After(endOfTime) {
		validUntil = endOfTime
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate serial number: %s", err)
	}

	host, err := os.Hostname()
	if err != nil {
		return nil, nil, err
	}

	ipAddresses := []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")}
	dnsNames := []string{host}
	if host != "localhost" {
		dnsNames = append(dnsNames, "localhost")
	}

	addIP := func(ipAddr net.IP) {
		for _, ip := range ipAddresses {
			if ip.Equal(ipAddr) {
				return
			}
		}
		ipAddresses = append(ipAddresses, ipAddr)
	}
	addHost := func(host string) {
		for _, dnsName := range dnsNames {
			if host == dnsName {
				return
			}
		}
		dnsNames = append(dnsNames, host)
	}

	addrs, err := interfaceAddrs()
	if err != nil {
		return nil, nil, err
	}
	for _, a := range addrs {
		ipAddr, _, err := net.ParseCIDR(a.String())
		if err == nil {
			addIP(ipAddr)
		}
	}

	for _, hostStr := range extraHosts {
		host, _, err := net.SplitHostPort(hostStr)
		if err != nil {
			host = hostStr
		}
		if ip := net.ParseIP(host); ip != nil {
			addIP(ip)
		} else {
			addHost(host)
		}
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{organization},
			CommonName:   host,
		},
		NotBefore: now.Add(-time.Hour * 24),
		NotAfter:  validUntil,

		KeyUsage: x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature |
			x509.KeyUsageCertSign,
		IsCA:                  true, // so can sign self.
		BasicConstraintsValid: true,

		DNSNames:    dnsNames,
		IPAddresses: ipAddresses,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template,
		&template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create certificate: %v", err)
	}

	certBuf := &bytes.Buffer{}
	err = pem.Encode(certBuf, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encode certificate: %v", err)
	}

	keybytes, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal private key: %v", err)
	}

	keyBuf := &bytes.Buffer{}
	err = pem.Encode(keyBuf, &pem.Block{Type: "PRIVATE KEY", Bytes: keybytes})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encode private key: %v", err)
	}

	return certBuf.Bytes(), keyBuf.Bytes(), nil
}

// WriteTLSCertPair generates a self-signed certificate/key pair via
// NewTLSCertPair, creates the parent directories of certFile and keyFile, and
// writes the PEM-encoded certificate (0644, public) and -- when writeKey is
// true -- the private key (0600).  On a key-write failure the certificate file
// is removed so a half-written pair is not left behind.  The parsed keypair is
// returned regardless of whether the key was persisted.
func WriteTLSCertPair(certFile, keyFile, org string, validUntil time.Time, extraHosts []string, writeKey bool) (tls.Certificate, error) {
	cert, key, err := NewTLSCertPair(org, validUntil, extraHosts)
	if err != nil {
		return tls.Certificate{}, err
	}

	keyPair, err := tls.X509KeyPair(cert, key)
	if err != nil {
		return tls.Certificate{}, err
	}

	if err := os.MkdirAll(filepath.Dir(certFile), 0700); err != nil {
		return tls.Certificate{}, err
	}
	if err := os.MkdirAll(filepath.Dir(keyFile), 0700); err != nil {
		return tls.Certificate{}, err
	}

	// The certificate is public, so 0644 is appropriate; the private key
	// must stay owner-only.
	if err := os.WriteFile(certFile, cert, 0644); err != nil {
		return tls.Certificate{}, err
	}
	if writeKey {
		// os.WriteFile keeps an existing file's mode, so remove any stale
		// key first to guarantee the new one is created owner-only (0600).
		if err := os.Remove(keyFile); err != nil && !os.IsNotExist(err) {
			// Don't leave the freshly written cert without a matching
			// key.
			_ = os.Remove(certFile)
			return tls.Certificate{}, err
		}
		if err := os.WriteFile(keyFile, key, 0600); err != nil {
			// Best-effort cleanup so a cert without its key is not
			// left behind.
			_ = os.Remove(certFile)
			return tls.Certificate{}, err
		}
	}

	return keyPair, nil
}
