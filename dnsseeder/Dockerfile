# Build stage
FROM golang:1.26.5-alpine AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build dnsseeder
WORKDIR /app/dnsseeder
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-s -w" -o /usr/local/bin/dnsseeder .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS and libcap for setcap
RUN apk --no-cache add ca-certificates libcap

WORKDIR /app

# Copy binary from builder
COPY --from=builder /usr/local/bin/dnsseeder /usr/local/bin/dnsseeder
COPY dnsseeder/entrypoint.sh /usr/local/bin/entrypoint.sh

# Set permissions
RUN chmod +x /usr/local/bin/entrypoint.sh

# Allow binding to low ports (like 53) without root (though container runs as root by default)
RUN setcap cap_net_bind_service=+ep /usr/local/bin/dnsseeder

# Create data directory
RUN mkdir -p /var/lib/dnsseeder

# Expose DNS port (TCP and UDP)
EXPOSE 53/udp 53/tcp

# Entrypoint
# Note: You must provide the -host flag when running the container
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
