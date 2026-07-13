# Build stage
FROM golang:1.26.1-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

WORKDIR /app/prlmon
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /usr/local/bin/prlmon .

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

COPY --from=builder /usr/local/bin/prlmon /usr/local/bin/prlmon

EXPOSE 9105

ENTRYPOINT ["/usr/local/bin/prlmon"]
