FROM golang:1.27 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /recall ./cmd/recall

FROM alpine:latest
RUN apk add --no-cache curl
COPY --from=builder /recall /recall
ENTRYPOINT ["/recall"]
