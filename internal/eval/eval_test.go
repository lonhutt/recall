package eval

import (
	"bytes"
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestLoadProbesRejectsInvalidProbes(t *testing.T) {
	cases := []struct {
		name, input, wantErr string
	}{
		{"empty array", `[]`, "no probes"},
		{"invalid json", `[{`, ""},
		{"missing want", `[{"query":"q","kind":"exact"}]`, "probe 0"},
		{"empty query", `[{"query":"q","want":["a"],"kind":"exact"},{"query":"","want":["a"],"kind":"exact"}]`, "probe 1"},
		{"unknown kind", `[{"query":"q","want":["a"],"kind":"vibes"}]`, "probe 0"},
		{"wrong field type", `[{"query":"q","want":["a"],"kind":"exact"},{"query":"q","want":"a","kind":"exact"}]`, "probe 1"},
		{"unknown field", `[{"query":"q","want":["a"],"kind":"exact","projct":"/p"}]`, "probe 0"},
		{"whitespace query", `[{"query":"  ","want":["a"],"kind":"exact"}]`, "probe 0"},
		{"trailing data", `[{"query":"q","want":["a"],"kind":"exact"}] [{"query":"r","want":["a"],"kind":"exact"}]`, "after"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := LoadProbes(strings.NewReader(c.input))
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err, c.wantErr)
			}
		})
	}
}

func TestLoadProbesAcceptsValidFile(t *testing.T) {
	probes, err := LoadProbes(strings.NewReader(`[
		{"query":"deploy failed with 401","want":["auth-note"],"kind":"situation"},
		{"query":"RECALL_HTTP_TOKEN","want":["env-note","other"],"project":"/p","kind":"exact"}
	]`))
	if err != nil {
		t.Fatalf("LoadProbes: %v", err)
	}
	want := []Probe{
		{Query: "deploy failed with 401", Want: []string{"auth-note"}, Kind: "situation"},
		{Query: "RECALL_HTTP_TOKEN", Want: []string{"env-note", "other"}, Project: "/p", Kind: "exact"},
	}
	if !reflect.DeepEqual(probes, want) {
		t.Errorf("probes = %+v, want %+v", probes, want)
	}
}

func fixedSearcher(slugs ...string) Searcher {
	return func(context.Context, Probe) ([]string, error) { return slugs, nil }
}

func TestRunRecordsRankOfFirstWantedSlug(t *testing.T) {
	probes := []Probe{
		{Query: "q1", Want: []string{"c"}},
		{Query: "q2", Want: []string{"x"}},
		{Query: "q3", Want: []string{"b", "a"}},
	}
	outcomes, err := Run(context.Background(), probes, fixedSearcher("a", "b", "c"))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	var ranks []int
	for _, o := range outcomes {
		ranks = append(ranks, o.Rank)
	}
	if !reflect.DeepEqual(ranks, []int{3, 0, 1}) {
		t.Errorf("ranks = %v, want [3 0 1]", ranks)
	}
}

func TestRunPropagatesSearchError(t *testing.T) {
	calls := 0
	search := func(context.Context, Probe) ([]string, error) {
		calls++
		if calls == 2 {
			return nil, errors.New("boom")
		}
		return nil, nil
	}
	probes := []Probe{{Query: "a", Want: []string{"x"}}, {Query: "b", Want: []string{"x"}}}
	_, err := Run(context.Background(), probes, search)
	if err == nil || !strings.Contains(err.Error(), "probe 1") {
		t.Errorf("err = %v, want an error naming probe 1", err)
	}
}

func outcomesWithRanks(ranks ...int) []Outcome {
	out := make([]Outcome, len(ranks))
	for i, r := range ranks {
		out[i] = Outcome{Probe: Probe{Query: string(rune('a' + i))}, Rank: r}
	}
	return out
}

func TestSummarizeCountsHitsAndMRR(t *testing.T) {
	s := Summarize(outcomesWithRanks(1, 3, 0, 2))
	if s.N != 4 || s.Hits1 != 1 || s.Hits3 != 3 || s.Hits10 != 3 {
		t.Errorf("summary = %+v, want N=4 Hits1=1 Hits3=3 Hits10=3", s)
	}
	wantMRR := (1 + 1.0/3 + 0 + 0.5) / 4
	if math.Abs(s.MRR-wantMRR) > 1e-9 {
		t.Errorf("MRR = %v, want %v", s.MRR, wantMRR)
	}
}

func TestDisagreementsPairsByProbe(t *testing.T) {
	a := outcomesWithRanks(1, 0, 4)
	b := outcomesWithRanks(0, 2, 1)
	onlyA, onlyB := Disagreements(a, b, 3)
	if !reflect.DeepEqual(onlyA, []Probe{a[0].Probe}) {
		t.Errorf("onlyA = %+v, want [p0]", onlyA)
	}
	if !reflect.DeepEqual(onlyB, []Probe{a[1].Probe, a[2].Probe}) {
		t.Errorf("onlyB = %+v, want [p1 p2]", onlyB)
	}
}

func TestUnknownWantsListsMissingSlugsOnce(t *testing.T) {
	probes := []Probe{{Want: []string{"a", "gone"}}, {Want: []string{"gone", "b"}}}
	exists := func(slug string) (bool, error) { return slug != "gone", nil }
	unknown, err := UnknownWants(probes, exists)
	if err != nil {
		t.Fatalf("UnknownWants: %v", err)
	}
	if !reflect.DeepEqual(unknown, []string{"gone"}) {
		t.Errorf("unknown = %v, want [gone]", unknown)
	}
}

func TestReportShowsBothModesAndDisagreements(t *testing.T) {
	probes := []Probe{
		{Query: "first situation", Want: []string{"x"}, Kind: "situation"},
		{Query: "RECALL_HTTP_TOKEN", Want: []string{"x"}, Kind: "exact"},
		{Query: "second situation", Want: []string{"x"}, Kind: "situation"},
	}
	mk := func(ranks ...int) []Outcome {
		out := make([]Outcome, len(ranks))
		for i, r := range ranks {
			out[i] = Outcome{Probe: probes[i], Rank: r}
		}
		return out
	}
	var buf bytes.Buffer
	err := Report(&buf, []Mode{
		{Name: "vector", Outcomes: mk(1, 0, 2)},
		{Name: "hybrid", Outcomes: mk(1, 1, 2)},
	})
	if err != nil {
		t.Fatalf("Report: %v", err)
	}
	out := buf.String()
	for _, want := range []string{"vector", "hybrid", "hits@3", "situation", "exact", "RECALL_HTTP_TOKEN"} {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q:\n%s", want, out)
		}
	}
}
