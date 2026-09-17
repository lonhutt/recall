package models

import "errors"

var (
	ErrNotFound      = errors.New("memory not found")
	ErrDuplicateSlug = errors.New("duplicate slug")
)
