//go:build integration

package postgres_test

import (
	"context"
	"testing"
	"time"

	"github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/lonhutt/recall/internal/store/postgres"
)

// newTestStore starts (once per test, cheap enough at this scale) a
// pgvector-enabled Postgres container, runs migrations against it, and
// returns a connected Store. Skips the test if Docker isn't reachable.
func newTestStore(t *testing.T) *postgres.Store {
	t.Helper()
	ctx := context.Background()

	ctr, err := tcpostgres.Run(ctx, "pgvector/pgvector:pg17",
		tcpostgres.WithDatabase("recall"),
		tcpostgres.WithUsername("recall"),
		tcpostgres.WithPassword("recall"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").WithOccurrence(2).WithStartupTimeout(30*time.Second),
		),
	)
	if err != nil {
		t.Skipf("skipping: could not start postgres container (is Docker running/accessible?): %v", err)
	}
	t.Cleanup(func() {
		if err := ctr.Terminate(context.Background()); err != nil {
			t.Logf("terminating postgres container: %v", err)
		}
	})

	connString, err := ctr.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("getting connection string: %v", err)
	}

	if err := postgres.RunMigrations(connString); err != nil {
		t.Fatalf("running migrations: %v", err)
	}

	store, err := postgres.New(ctx, connString)
	if err != nil {
		t.Fatalf("connecting store: %v", err)
	}
	t.Cleanup(store.Close)

	return store
}
