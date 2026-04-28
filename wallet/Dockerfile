FROM golang:1.26.2-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o wallet ./wallet

FROM alpine:latest
RUN adduser -D -s /bin/sh wallet
COPY --from=builder /app/wallet/wallet /usr/local/bin/wallet

RUN mkdir -p /home/wallet/.wallet && \
    chown -R wallet:wallet /home/wallet

USER wallet

WORKDIR /home/wallet


EXPOSE 44207

ENTRYPOINT ["wallet"]

CMD ["--help"]
