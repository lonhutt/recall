// Package httpauth provides simple bearer-token authentication for Recall's
// HTTP transport.
package httpauth

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// Bearer wraps next, requiring an "Authorization: Bearer <token>" header
// matching token on every request.
func Bearer(token string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
