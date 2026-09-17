.PHONY: up down migrate-up migrate-down run test test-integration build

up:
	docker compose up -d --build

down:
	docker compose down

migrate-up:
	go run ./cmd/recall migrate

run:
	go run ./cmd/recall

build:
	go build -o bin/recall ./cmd/recall

test:
	go test ./...

test-integration:
	go test -tags=integration ./...
