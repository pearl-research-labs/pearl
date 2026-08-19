# syntax=docker/dockerfile:1

ARG GO_VERSION=1.26.6
ARG ALPINE_VERSION=3.24

FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder

WORKDIR /pearl

# Warm the module cache in its own layer so source edits don't re-download.
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY . .

# GIT_COMMIT stamps `coredns -version` output; CI passes the build SHA.
# CoreDNS version and plugin set are defined by go.mod and
# coredns-dnsseed/cmd/coredns; -s -w matches upstream's stripped releases.
ARG GIT_COMMIT=""
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath \
    -ldflags="-s -w -X github.com/coredns/coredns/coremain.GitCommit=${GIT_COMMIT}" \
    -o /out/coredns ./coredns-dnsseed/cmd/coredns

# ---

# scratch keeps the runtime surface to the static binary and CA bundle,
# mirroring upstream CoreDNS images. There is no shell: debug with an
# ephemeral container (kubectl debug). The image binds only unprivileged
# ports (:1053 by convention); map to :53 at the Service.
FROM scratch

LABEL maintainer="Pearl Research Labs"
LABEL org.opencontainers.image.source="https://github.com/pearl-research-labs/pearl"
LABEL org.opencontainers.image.description="Pearl DNS Seeder - CoreDNS with dnsseed plugin"

COPY --from=builder /etc/ssl/certs /etc/ssl/certs
COPY --from=builder /out/coredns /coredns

USER 65534:65534

EXPOSE 1053 1053/udp

# No Docker HEALTHCHECK: Kubernetes ignores it. Use the CoreDNS `health`
# (liveness) and `ready` (readiness) plugin endpoints as probes instead.

ENTRYPOINT ["/coredns"]
CMD ["-conf", "/etc/coredns/Corefile"]
