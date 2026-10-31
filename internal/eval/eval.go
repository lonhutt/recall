// Package eval scores search ranking against a fixed set of probe queries.
package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
)

const (
	KindSituation = "situation"
	KindExact     = "exact"
)

// Probe is one query with the slugs that count as a correct answer to it.
type Probe struct {
	Query   string   `json:"query"`
	Want    []string `json:"want"`
	Project string   `json:"project,omitempty"`
	Kind    string   `json:"kind"`
}

// Searcher returns ranked slugs for a probe, best first.
type Searcher func(ctx context.Context, p Probe) ([]string, error)

// Outcome is where a probe's first wanted slug ranked; Rank is 1-based, 0
// when none of them came back.
type Outcome struct {
	Probe Probe
	Rank  int
}

type Summary struct {
	N, Hits1, Hits3, Hits10 int
	MRR                     float64
}

// Mode is one search configuration's outcomes, for Report.
type Mode struct {
	Name     string
	Outcomes []Outcome
}

// LoadProbes reads a JSON array of probes, rejecting anything it doesn't
// fully understand: a typo'd field silently changing a probe would skew the
// comparison without failing it.
func LoadProbes(r io.Reader) ([]Probe, error) {
	dec := json.NewDecoder(r)
	var raw []json.RawMessage
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("decoding probes: %w", err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after the probe array")
	}
	if len(raw) == 0 {
		return nil, errors.New("no probes")
	}
	probes := make([]Probe, len(raw))
	for i, msg := range raw {
		pd := json.NewDecoder(bytes.NewReader(msg))
		pd.DisallowUnknownFields()
		if err := pd.Decode(&probes[i]); err != nil {
			return nil, fmt.Errorf("probe %d: %w", i, err)
		}
	}
	for i, p := range probes {
		switch {
		case strings.TrimSpace(p.Query) == "":
			return nil, fmt.Errorf("probe %d: query is required", i)
		case len(p.Want) == 0:
			return nil, fmt.Errorf("probe %d (%q): want needs at least one slug", i, p.Query)
		case p.Kind != KindSituation && p.Kind != KindExact:
			return nil, fmt.Errorf("probe %d (%q): kind must be %q or %q, got %q", i, p.Query, KindSituation, KindExact, p.Kind)
		}
	}
	return probes, nil
}

// UnknownWants lists wanted slugs that don't exist. A probe pointing at a
// renamed or deleted memory is a permanent miss in every mode, which hides
// real differences between them.
func UnknownWants(probes []Probe, exists func(slug string) (bool, error)) ([]string, error) {
	seen := map[string]bool{}
	var unknown []string
	for _, p := range probes {
		for _, slug := range p.Want {
			if seen[slug] {
				continue
			}
			seen[slug] = true
			ok, err := exists(slug)
			if err != nil {
				return nil, fmt.Errorf("checking slug %q: %w", slug, err)
			}
			if !ok {
				unknown = append(unknown, slug)
			}
		}
	}
	return unknown, nil
}

func Run(ctx context.Context, probes []Probe, search Searcher) ([]Outcome, error) {
	outcomes := make([]Outcome, len(probes))
	for i, p := range probes {
		slugs, err := search(ctx, p)
		if err != nil {
			return nil, fmt.Errorf("probe %d (%q): %w", i, p.Query, err)
		}
		outcomes[i] = Outcome{Probe: p, Rank: rankOf(slugs, p.Want)}
	}
	return outcomes, nil
}

func rankOf(slugs, want []string) int {
	for i, s := range slugs {
		if slices.Contains(want, s) {
			return i + 1
		}
	}
	return 0
}

func hit(o Outcome, k int) bool { return o.Rank >= 1 && o.Rank <= k }

func Summarize(outcomes []Outcome) Summary {
	s := Summary{N: len(outcomes)}
	var rr float64
	for _, o := range outcomes {
		if hit(o, 1) {
			s.Hits1++
		}
		if hit(o, 3) {
			s.Hits3++
		}
		if hit(o, 10) {
			s.Hits10++
		}
		if o.Rank > 0 {
			rr += 1 / float64(o.Rank)
		}
	}
	if s.N > 0 {
		s.MRR = rr / float64(s.N)
	}
	return s
}

// Disagreements pairs a and b by probe index and returns the probes only one
// side hit within the top k.
func Disagreements(a, b []Outcome, k int) (onlyA, onlyB []Probe) {
	for i := range a {
		ha, hb := hit(a[i], k), hit(b[i], k)
		switch {
		case ha && !hb:
			onlyA = append(onlyA, a[i].Probe)
		case hb && !ha:
			onlyB = append(onlyB, a[i].Probe)
		}
	}
	return onlyA, onlyB
}

func Report(w io.Writer, modes []Mode) error {
	pw := &printer{w: w}
	for _, m := range modes {
		pw.printf("== %s\n", m.Name)
		pw.summary("all", Summarize(m.Outcomes))
		for _, kind := range []string{KindSituation, KindExact} {
			var subset []Outcome
			for _, o := range m.Outcomes {
				if o.Probe.Kind == kind {
					subset = append(subset, o)
				}
			}
			if len(subset) > 0 {
				pw.summary(kind, Summarize(subset))
			}
		}
	}
	if len(modes) == 2 {
		onlyA, onlyB := Disagreements(modes[0].Outcomes, modes[1].Outcomes, 3)
		pw.printf("== disagreements at hits@3\n")
		for _, d := range []struct {
			name   string
			probes []Probe
		}{{modes[0].Name, onlyA}, {modes[1].Name, onlyB}} {
			pw.printf("only %s (%d):\n", d.name, len(d.probes))
			for _, p := range d.probes {
				pw.printf("  [%s] %s -> %v\n", p.Kind, p.Query, p.Want)
			}
		}
	}
	return pw.err
}

// printer keeps the first write error so Report can stay a flat list of
// prints.
type printer struct {
	w   io.Writer
	err error
}

func (p *printer) printf(format string, args ...any) {
	if p.err == nil {
		_, p.err = fmt.Fprintf(p.w, format, args...)
	}
}

func (p *printer) summary(label string, s Summary) {
	p.printf("  %-9s n=%-3d hits@1 %s  hits@3 %s  hits@10 %s  MRR %.3f\n",
		label, s.N, frac(s.Hits1, s.N), frac(s.Hits3, s.N), frac(s.Hits10, s.N), s.MRR)
}

func frac(k, n int) string {
	if n == 0 {
		return "0/0"
	}
	return fmt.Sprintf("%d/%d (%.0f%%)", k, n, 100*float64(k)/float64(n))
}
