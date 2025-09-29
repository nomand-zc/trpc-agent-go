//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//

// Package summary provides session summarization functionality for trpc-agent-go.
// It includes automatic conversation compression, LLM integration, and configurable
// trigger conditions to reduce memory usage while maintaining conversation context.
package summary

import (
	"context"
	"time"

	"trpc.group/trpc-go/trpc-agent-go/session"
)

// SessionSummarizer defines the interface for generating session summaries.
type SessionSummarizer interface {
	// ShouldSummarize checks if the session should be summarized.
	ShouldSummarize(sess *session.Session) bool

	// Summarize generates a summary without modifying the session events.
	Summarize(ctx context.Context, sess *session.Session) (string, error)

	// Metadata returns metadata about the summarizer configuration.
	Metadata() map[string]any
}

// SessionSummary represents a summary of a session's conversation history.
type SessionSummary struct {
	// ID is the ID of the session.
	ID string `json:"id"`
	// Summary is the summary of the session.
	Summary string `json:"summary"`
	// CreatedAt is the time the summary was created.
	CreatedAt time.Time `json:"created_at"`
	// Metadata is the metadata of the summary.
	Metadata map[string]any `json:"metadata"`
}
