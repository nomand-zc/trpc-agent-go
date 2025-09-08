//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//
//

package processor

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/tool"
)

// mockModel implements model.Model for testing
type mockModel struct {
	ShouldError bool
	responses   []*model.Response
	currentIdx  int
}

func (m *mockModel) Info() model.Info {
	return model.Info{
		Name: "mock",
	}
}

func (m *mockModel) GenerateContent(ctx context.Context, req *model.Request) (<-chan *model.Response, error) {
	if m.ShouldError {
		return nil, errors.New("mock model error")
	}

	respChan := make(chan *model.Response, len(m.responses))

	go func() {
		defer close(respChan)
		for _, resp := range m.responses {
			select {
			case respChan <- resp:
			case <-ctx.Done():
				return
			}
		}
	}()

	return respChan, nil
}

// Minimal callable tool used by tests above
type mockCallableTool struct {
	declaration *tool.Declaration
	callFn      func(ctx context.Context, args []byte) (any, error)
}

func (m *mockCallableTool) Declaration() *tool.Declaration { return m.declaration }
func (m *mockCallableTool) Call(ctx context.Context, args []byte) (any, error) {
	return m.callFn(ctx, args)
}

func TestExecuteToolCall_MapsSubAgentToTransfer(t *testing.T) {
	ctx := context.Background()
	p := NewFunctionCallResponseProcessor()

	// Prepare invocation with a parent agent that has a sub-agent named weather-agent.
	inv := &agent.Invocation{
		AgentName: "weather-agent",
	}

	// Prepare tools: only transfer tool is exposed, no weather-agent tool.
	tools := map[string]tool.Tool{
		"weather-agent": &mockCallableTool{
			declaration: &tool.Declaration{Name: "weather-agent", Description: "transfer"},
			callFn: func(_ context.Context, args []byte) (any, error) {
				return "Tokyo'weather is good", nil
			},
		},
	}

	// Original tool call uses sub-agent name directly.
	originalArgs := []byte(`{"message":"What's the weather like in Tokyo?"}`)
	pc := model.ToolCall{
		ID: "call-1",
		Function: model.FunctionDefinitionParam{
			Name:      "weather-agent",
			Arguments: originalArgs,
		},
	}

	choice, err := p.executeToolCall(ctx, inv, pc, tools, 0)
	res, _ := json.Marshal("Tokyo'weather is good")
	require.NoError(t, err)
	require.NotNil(t, choice)
	assert.Equal(t, string(res), choice.Message.Content)
}

func TestExecuteToolCall_ToolNotFound_ReturnsErrorChoice(t *testing.T) {
	ctx := context.Background()
	p := NewFunctionCallResponseProcessor()

	// Invocation without matching sub-agent and with a mock model to satisfy logging.
	inv := &agent.Invocation{
		Model: &mockModel{},
	}

	tools := map[string]tool.Tool{} // No tools available.

	pc2 := model.ToolCall{
		ID: "call-404",
		Function: model.FunctionDefinitionParam{
			Name:      "non-existent-tool",
			Arguments: []byte(`{}`),
		},
	}

	choice, err := p.executeToolCall(ctx, inv, pc2, tools, 0)
	require.NoError(t, err)
	require.NotNil(t, choice)
	assert.Equal(t, ErrorToolNotFound, choice.Message.Content)
	assert.Equal(t, "call-404", choice.Message.ToolID)
}
