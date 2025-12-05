//
// Tencent is pleased to support the open source community by making trpc-agent-go available.
//
// Copyright (C) 2025 Tencent.  All rights reserved.
//
// trpc-agent-go is licensed under the Apache License Version 2.0.
//
//

// Package llmagent provides an LLM agent implementation.
package llmagent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/trace"

	"trpc.group/trpc-go/trpc-agent-go/agent"
	"trpc.group/trpc-go/trpc-agent-go/codeexecutor"
	localexec "trpc.group/trpc-go/trpc-agent-go/codeexecutor/local"
	"trpc.group/trpc-go/trpc-agent-go/event"
	"trpc.group/trpc-go/trpc-agent-go/internal/flow"
	"trpc.group/trpc-go/trpc-agent-go/internal/flow/llmflow"
	"trpc.group/trpc-go/trpc-agent-go/internal/flow/processor"
	itelemetry "trpc.group/trpc-go/trpc-agent-go/internal/telemetry"
	itool "trpc.group/trpc-go/trpc-agent-go/internal/tool"
	knowledgetool "trpc.group/trpc-go/trpc-agent-go/knowledge/tool"
	"trpc.group/trpc-go/trpc-agent-go/model"
	"trpc.group/trpc-go/trpc-agent-go/planner"
	"trpc.group/trpc-go/trpc-agent-go/telemetry/trace"
	"trpc.group/trpc-go/trpc-agent-go/tool"
	toolskill "trpc.group/trpc-go/trpc-agent-go/tool/skill"
	"trpc.group/trpc-go/trpc-agent-go/tool/transfer"
)

// localruntimeFallback returns a simple local workspace executor used when
// no explicit executor is provided.
func defaultCodeExecutor() codeexecutor.CodeExecutor {
	return localexec.New()
}

// LLMAgent is an agent that uses an LLM to generate responses.
type LLMAgent struct {
	name          string
	mu            sync.RWMutex
	model         model.Model
	models        map[string]model.Model // Registered models for switching
	description   string
	instruction   string
	systemPrompt  string
	genConfig     model.GenerationConfig
	flow          flow.Flow
	tools         []tool.Tool     // All tools (user tools + framework tools)
	userToolNames map[string]bool // Names of tools explicitly registered
	// via WithTools and WithToolSets.
	codeExecutor         codeexecutor.CodeExecutor
	planner              planner.Planner
	subAgents            []agent.Agent // Sub-agents that can be delegated to
	agentCallbacks       *agent.Callbacks
	outputKey            string         // Key to store output in session state
	outputSchema         map[string]any // JSON schema for output validation
	inputSchema          map[string]any // JSON schema for input validation
	structuredOutput     *model.StructuredOutput
	structuredOutputType reflect.Type
	option               Options
}

// New creates a new LLMAgent with the given options.
func New(name string, opts ...Option) *LLMAgent {
	options := defaultOptions
	for _, opt := range opts {
		opt(&options)
	}

	// Validate output_schema configuration before registering tools.
	if options.OutputSchema != nil {
		if len(options.Tools) > 0 || len(options.ToolSets) > 0 {
			panic("Invalid LLMAgent configuration: if output_schema is set, tools and toolSets must be empty")
		}
		if options.Knowledge != nil {
			panic("Invalid LLMAgent configuration: if output_schema is set, knowledge must be empty")
		}
		if len(options.SubAgents) > 0 {
			panic("Invalid LLMAgent configuration: if output_schema is set, sub_agents must be empty to disable agent transfer")
		}
	}

	// Construct the agent first so request processors can access dynamic getters.
	a := &LLMAgent{
		name:                 name,
		model:                options.Model,
		models:               make(map[string]model.Model),
		description:          options.Description,
		instruction:          options.Instruction,
		systemPrompt:         options.GlobalInstruction,
		genConfig:            options.GenerationConfig,
		codeExecutor:         options.codeExecutor,
		planner:              options.Planner,
		subAgents:            options.SubAgents,
		agentCallbacks:       options.AgentCallbacks,
		outputKey:            options.OutputKey,
		outputSchema:         options.OutputSchema,
		inputSchema:          options.InputSchema,
		structuredOutput:     options.StructuredOutput,
		structuredOutputType: options.StructuredOutputType,
		option:               options,
	}

	// Initialize models map and determine the initial model.
	a.initializeModels()

	// load tools if needed
	if !options.RefreshToolSetsOnRun {
		a.registerTools()
	}

	// Prepare request processors in the correct order, wiring dynamic getters.
	requestProcessors := a.buildRequestProcessors()
	responseProcessors := a.buildResponseProcessors()

	// Create flow with the provided processors and options.
	flowOpts := llmflow.Options{
		ChannelBufferSize: options.ChannelBufferSize,
		ModelCallbacks:    options.ModelCallbacks,
	}

	a.flow = llmflow.New(
		requestProcessors, responseProcessors,
		flowOpts,
	)

	return a
}

func (a *LLMAgent) buildResponseProcessors() []flow.ResponseProcessor {
	// Prepare response processors.
	var responseProcessors []flow.ResponseProcessor

	// Add planning response processor if planner is configured.
	if a.option.Planner != nil {
		planningResponseProcessor := processor.NewPlanningResponseProcessor(a.option.Planner)
		responseProcessors = append(responseProcessors, planningResponseProcessor)
	}

	responseProcessors = append(responseProcessors, processor.NewCodeExecutionResponseProcessor())

	// Add output response processor if output_key or output_schema is configured or structured output is requested.
	if a.option.OutputKey != "" || a.option.OutputSchema != nil || a.option.StructuredOutput != nil {
		orp := processor.NewOutputResponseProcessor(a.option.OutputKey, a.option.OutputSchema)
		responseProcessors = append(responseProcessors, orp)
	}

	toolcallProcessor := processor.NewFunctionCallResponseProcessor(a.option.EnableParallelTools, a.option.ToolCallbacks)
	// Configure default transfer message for direct sub-agent calls.
	// Default behavior (when not configured): enabled with built-in default message.
	if a.option.DefaultTransferMessage != nil {
		// Explicitly configured via WithDefaultTransferMessage.
		processor.SetDefaultTransferMessage(*a.option.DefaultTransferMessage)
	}
	responseProcessors = append(responseProcessors, toolcallProcessor)

	// Add transfer response processor if sub-agents are configured.
	if len(a.option.SubAgents) > 0 {
		transferResponseProcessor := processor.NewTransferResponseProcessor(a.option.EndInvocationAfterTransfer)
		responseProcessors = append(responseProcessors, transferResponseProcessor)
	}

	return responseProcessors
}

// buildRequestProcessors constructs the request processors in the required order.
func (a *LLMAgent) buildRequestProcessors() []flow.RequestProcessor {
	var requestProcessors []flow.RequestProcessor

	// 1. Basic processor - handles generation config.
	basicOptions := []processor.BasicOption{
		processor.WithGenerationConfig(a.option.GenerationConfig),
	}
	basicProcessor := processor.NewBasicRequestProcessor(basicOptions...)
	requestProcessors = append(requestProcessors, basicProcessor)

	// 2. Planning processor - handles planning instructions if planner is configured.
	if a.option.Planner != nil {
		planningProcessor := processor.NewPlanningRequestProcessor(a.option.Planner)
		requestProcessors = append(requestProcessors, planningProcessor)
	}

	// 3. Instruction processor - adds instruction content and system prompt.
	if a.option.Instruction != "" || a.option.GlobalInstruction != "" ||
		(a.option.StructuredOutput != nil && a.option.StructuredOutput.JSONSchema != nil) {
		instructionOpts := []processor.InstructionRequestProcessorOption{
			processor.WithOutputSchema(a.option.OutputSchema),
		}
		// Fallback injection for structured output when the provider doesn't enforce JSON Schema natively.
		if a.option.StructuredOutput != nil && a.option.StructuredOutput.JSONSchema != nil {
			instructionOpts = append(instructionOpts,
				processor.WithStructuredOutputSchema(a.option.StructuredOutput.JSONSchema.Schema),
			)
		}
		// Always wire dynamic getters so instructions can be updated at runtime.
		instructionOpts = append(instructionOpts,
			processor.WithInstructionGetter(func() string { return a.getInstruction() }),
			processor.WithSystemPromptGetter(func() string { return a.getSystemPrompt() }),
		)
		instructionProcessor := processor.NewInstructionRequestProcessor(
			"", // static value unused when getters are present
			"", // static value unused when getters are present
			instructionOpts...,
		)
		requestProcessors = append(requestProcessors, instructionProcessor)
	}

	// 4. Identity processor - sets agent identity.
	if a.name != "" || a.option.Description != "" {
		identityProcessor := processor.NewIdentityRequestProcessor(
			a.name,
			a.option.Description,
			processor.WithAddNameToInstruction(a.option.AddNameToInstruction),
		)
		requestProcessors = append(requestProcessors, identityProcessor)
	}

	// 5. Time processor - adds current time information if enabled.
	if a.option.AddCurrentTime {
		timeProcessor := processor.NewTimeRequestProcessor(
			processor.WithAddCurrentTime(true),
			processor.WithTimezone(a.option.Timezone),
			processor.WithTimeFormat(a.option.TimeFormat),
		)
		requestProcessors = append(requestProcessors, timeProcessor)
	}

	// 6. Skills processor - injects skill overview and loaded contents
	// when a skills repository is configured. This ensures the model
	// sees available skills (names/descriptions) and any loaded
	// SKILL.md/doc texts before deciding on tool calls.
	if a.option.SkillsRepository != nil {
		skillsProcessor := processor.NewSkillsRequestProcessor(
			a.option.SkillsRepository,
		)
		requestProcessors = append(requestProcessors, skillsProcessor)
	}

	// 7. Content processor - appends conversation/context history.
	contentProcessor := processor.NewContentRequestProcessor(
		processor.WithAddContextPrefix(a.option.AddContextPrefix),
		processor.WithAddSessionSummary(a.option.AddSessionSummary),
		processor.WithMaxHistoryRuns(a.option.MaxHistoryRuns),
		processor.WithPreserveSameBranch(a.option.PreserveSameBranch),
		processor.WithTimelineFilterMode(a.option.messageTimelineFilterMode),
		processor.WithBranchFilterMode(a.option.messageBranchFilterMode),
	)
	requestProcessors = append(requestProcessors, contentProcessor)

	return requestProcessors
}

// initializeModels initializes the models map and determines the initial
// model based on WithModel and WithModels options.
func (a *LLMAgent) initializeModels() {
	// Case 1: No models configured at all.
	if a.option.Model == nil && len(a.option.Models) == 0 {
		return
	}

	// Case 2: Only WithModel is set, no WithModels.
	if len(a.option.Models) == 0 {
		a.models[defaultModelName] = a.option.Model
		return
	}

	a.models = a.option.Models
	// If WithModel is also set, use it as the initial model.
	if a.option.Model != nil {
		// Check if the model is already in the models map.
		found := false
		for _, m := range a.option.Models {
			if m == a.option.Model {
				found = true
				break
			}
		}
		// If not found, add it with the default name.
		if !found {
			a.models[defaultModelName] = a.option.Model
		}
		return
	}

	// WithModels is set but WithModel is not, use the first model from map.
	// Note: map iteration order is not guaranteed.
	for _, m := range a.option.Models {
		a.model = m
		break
	}
}

func (a *LLMAgent) registerTools() {
	// Track user-registered tool names from WithTools and WithToolSets.
	// These are tools explicitly registered by the user and can be subject to filtering.
	userToolNames := make(map[string]bool)

	// Tools from WithTools are user tools.
	for _, t := range a.option.Tools {
		userToolNames[t.Declaration().Name] = true
	}

	// Start with direct tools.
	allTools := make([]tool.Tool, 0, len(a.option.Tools))
	allTools = append(allTools, a.option.Tools...)

	for _, toolSet := range a.option.ToolSets {
		// Create named toolset wrapper to avoid name conflicts.
		namedToolSet := itool.NewNamedToolSet(toolSet)
		setTools := namedToolSet.Tools(context.Background())
		for _, t := range setTools {
			allTools = append(allTools, t)
			// Mark toolset tools as user tools.
			userToolNames[t.Declaration().Name] = true
		}
	}

	// Add knowledge search tool if knowledge base is provided.
	// This is a FRAMEWORK tool (auto-added by framework), NOT a user tool.
	// It should never be filtered out by user tool filters.
	if a.option.Knowledge != nil {
		toolOpts := []knowledgetool.Option{
			knowledgetool.WithFilter(a.option.KnowledgeFilter),
		}
		if a.option.KnowledgeConditionedFilter != nil {
			toolOpts = append(toolOpts, knowledgetool.WithConditionedFilter(a.option.KnowledgeConditionedFilter))
		}

		if a.option.EnableKnowledgeAgenticFilter {
			agenticKnowledge := knowledgetool.NewAgenticFilterSearchTool(
				a.option.Knowledge, a.option.AgenticFilterInfo, toolOpts...,
			)
			allTools = append(allTools, agenticKnowledge)
			// Do NOT add to userToolNames - this is a framework tool.
		} else {
			knowledgeTool := knowledgetool.NewKnowledgeSearchTool(
				a.option.Knowledge, toolOpts...,
			)
			allTools = append(allTools, knowledgeTool)
			// Do NOT add to userToolNames - this is a framework tool.
		}
	}

	// Add skill tools when skills are enabled.
	if a.option.SkillsRepository != nil {
		allTools = append(allTools,
			toolskill.NewLoadTool(a.option.SkillsRepository))
		// Specialized doc tools for clarity and control.
		allTools = append(allTools,
			toolskill.NewSelectDocsTool(a.option.SkillsRepository))
		allTools = append(allTools,
			toolskill.NewListDocsTool(a.option.SkillsRepository))
		// Provide executor to skill_run, fallback to local.
		exec := a.option.codeExecutor
		if exec == nil {
			exec = defaultCodeExecutor()
		}
		allTools = append(allTools,
			toolskill.NewRunTool(a.option.SkillsRepository, exec))
	}

	// Add transfer tool
	if len(a.subAgents) > 0 {
		agentInfos := make([]agent.Info, len(a.subAgents))
		for i, subAgent := range a.subAgents {
			agentInfos[i] = subAgent.Info()
		}

		transferTool := transfer.New(agentInfos)
		allTools = append(allTools, transferTool)
	}

	a.tools = allTools
	a.userToolNames = userToolNames
}

// Run implements the agent.Agent interface.
// It executes the LLM agent flow and returns a channel of events.
func (a *LLMAgent) Run(ctx context.Context, invocation *agent.Invocation) (e <-chan *event.Event, err error) {
	a.setupInvocation(invocation)

	ctx, span := trace.Tracer.Start(ctx, fmt.Sprintf("%s %s", itelemetry.OperationInvokeAgent, a.name))
	itelemetry.TraceBeforeInvokeAgent(span, invocation, a.description, a.systemPrompt+a.instruction, &a.genConfig)

	ctx, flowEventChan, err := a.executeAgentFlow(ctx, invocation)
	if err != nil {
		// Check if this is a custom response error (early return)
		var customErr *haveCustomResponseError
		if errors.As(err, &customErr) {
			span.End()
			return customErr.EventChan, nil
		}
		// Handle actual errors
		span.SetStatus(codes.Error, err.Error())
		span.SetAttributes(attribute.String(itelemetry.KeyErrorType, itelemetry.ValueDefaultErrorType))
		span.End()
		return nil, err
	}

	return a.wrapEventChannel(ctx, invocation, flowEventChan, span), nil
}

// executeAgentFlow executes the agent flow with before agent callbacks.
// Returns the updated context, event channel, and any error that occurred.
func (a *LLMAgent) executeAgentFlow(ctx context.Context, invocation *agent.Invocation) (context.Context, <-chan *event.Event, error) {
	if a.agentCallbacks != nil {
		result, err := a.agentCallbacks.RunBeforeAgent(ctx, &agent.BeforeAgentArgs{
			Invocation: invocation,
		})
		if err != nil {
			return ctx, nil, fmt.Errorf("before agent callback failed: %w", err)
		}
		// Use the context from result if provided.
		if result != nil && result.Context != nil {
			ctx = result.Context
		}
		if result != nil && result.CustomResponse != nil {
			// Create a channel that returns the custom response and then closes.
			eventChan := make(chan *event.Event, 1)
			// Create an event from the custom response.
			customEvent := event.NewResponseEvent(invocation.InvocationID, invocation.AgentName, result.CustomResponse)
			agent.EmitEvent(ctx, invocation, eventChan, customEvent)
			close(eventChan)
			return ctx, nil, &haveCustomResponseError{EventChan: eventChan}
		}
	}

	// Use the underlying flow to execute the agent logic.
	flowEventChan, err := a.flow.Run(ctx, invocation)
	if err != nil {
		return ctx, nil, err
	}

	return ctx, flowEventChan, nil

}

// haveCustomResponseError represents an early return due to a custom response from before agent callbacks.
// This is not an actual error but a signal to return early with the custom response.
type haveCustomResponseError struct {
	EventChan <-chan *event.Event
}

func (e *haveCustomResponseError) Error() string {
	return "custom response provided, returning early"
}

// setupInvocation sets up the invocation
func (a *LLMAgent) setupInvocation(invocation *agent.Invocation) {
	// Set model: prioritize RunOptions.Model, then RunOptions.ModelName, then agent's default model.
	a.mu.RLock()
	// Check if a per-request model is specified.
	if invocation.RunOptions.Model != nil {
		// Use the model directly from RunOptions.
		invocation.Model = invocation.RunOptions.Model
	} else if invocation.RunOptions.ModelName != "" {
		// Look up model by name from registered models.
		if m, ok := a.models[invocation.RunOptions.ModelName]; ok {
			invocation.Model = m
		} else {
			// If model name not found, fall back to agent's default model.
			// Log a warning but don't fail the request.
			invocation.Model = a.model
		}
	} else {
		// Use agent's default model.
		invocation.Model = a.model
	}
	a.mu.RUnlock()

	// Set agent and agent name
	invocation.Agent = a
	invocation.AgentName = a.name

	// Propagate structured output configuration into invocation and request path.
	invocation.StructuredOutputType = a.structuredOutputType
	invocation.StructuredOutput = a.structuredOutput

	// load tools if needed
	if a.option.RefreshToolSetsOnRun {
		a.registerTools()
	}
	invocation.SetState(llmflow.StateKeyToolsSnapshot, &agent.CacheTools{
		Tools:         a.tools,
		UserToolNames: a.userToolNames,
	})
}

// wrapEventChannel wraps the event channel to apply after agent callbacks.
func (a *LLMAgent) wrapEventChannel(
	ctx context.Context,
	invocation *agent.Invocation,
	originalChan <-chan *event.Event,
	span sdktrace.Span,
) <-chan *event.Event {
	// Create a new channel with the same capacity as the original channel
	wrappedChan := make(chan *event.Event, cap(originalChan))

	go func() {
		var fullRespEvent *event.Event
		tokenUsage := &itelemetry.TokenUsage{}
		defer func() {
			if fullRespEvent != nil {
				itelemetry.TraceAfterInvokeAgent(span, fullRespEvent, tokenUsage)
			}
			span.End()
			close(wrappedChan)
		}()

		// Forward all events from the original channel
		for evt := range originalChan {
			if evt != nil && evt.Response != nil {

				if evt.Response.Usage != nil {
					tokenUsage.PromptTokens = evt.Response.Usage.PromptTokens
					tokenUsage.CompletionTokens = evt.Response.Usage.CompletionTokens
					tokenUsage.TotalTokens = evt.Response.Usage.TotalTokens
				}
				if !evt.Response.IsPartial {
					fullRespEvent = evt
				}

			}
			if err := event.EmitEvent(ctx, wrappedChan, evt); err != nil {
				return
			}
		}

		// Collect error from the final response event.
		var agentErr error
		if fullRespEvent != nil && fullRespEvent.Response != nil && fullRespEvent.Response.Error != nil {
			agentErr = fmt.Errorf("%s: %s", fullRespEvent.Response.Error.Type, fullRespEvent.Response.Error.Message)
		}

		// After all events are processed, run after agent callbacks
		if a.agentCallbacks != nil {
			result, err := a.agentCallbacks.RunAfterAgent(ctx, &agent.AfterAgentArgs{
				Invocation:        invocation,
				Error:             agentErr,
				FullResponseEvent: fullRespEvent,
			})
			// Use the context from result if provided.
			if result != nil && result.Context != nil {
				ctx = result.Context
			}
			var evt *event.Event
			if err != nil {
				// Send error event.
				evt = event.NewErrorEvent(
					invocation.InvocationID,
					invocation.AgentName,
					agent.ErrorTypeAgentCallbackError,
					err.Error(),
				)
			} else if result != nil && result.CustomResponse != nil {
				// Create an event from the custom response.
				evt = event.NewResponseEvent(invocation.InvocationID, invocation.AgentName, result.CustomResponse)
			}
			if evt != nil {
				fullRespEvent = evt
			}

			agent.EmitEvent(ctx, invocation, wrappedChan, evt)
		}
	}()

	return wrappedChan
}

// Info implements the agent.Agent interface.
// It returns the basic information about this agent.
func (a *LLMAgent) Info() agent.Info {
	return agent.Info{
		Name:         a.name,
		Description:  a.description,
		InputSchema:  a.inputSchema,
		OutputSchema: a.outputSchema,
	}
}

// Tools implements the agent.Agent interface.
// It returns the list of tools available to the agent, including
// transfer tools.
func (a *LLMAgent) Tools() []tool.Tool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	// load the latest tool list
	if a.option.RefreshToolSetsOnRun {
		a.registerTools()
	}

	base := make([]tool.Tool, len(a.tools))
	copy(base, a.tools)

	return base
}

// SubAgents returns the list of sub-agents for this agent.
func (a *LLMAgent) SubAgents() []agent.Agent {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if len(a.subAgents) == 0 {
		return nil
	}

	subAgents := make([]agent.Agent, len(a.subAgents))
	copy(subAgents, a.subAgents)
	return subAgents
}

// FindSubAgent finds a sub-agent by name.
// Returns nil if no sub-agent with the given name is found.
func (a *LLMAgent) FindSubAgent(name string) agent.Agent {
	a.mu.RLock()
	defer a.mu.RUnlock()

	for _, subAgent := range a.subAgents {
		if subAgent.Info().Name == name {
			return subAgent
		}
	}
	return nil
}

// UserTools returns the list of tools that were explicitly registered
// by the user via WithTools and WithToolSets options.
//
// User tools (can be filtered):
//   - Tools registered via WithTools
//   - Tools registered via WithToolSets
//
// Framework tools (never filtered, not included in this list):
//   - knowledge_search / agentic_knowledge_search (auto-added when
//     WithKnowledge is set)
//   - transfer_to_agent (auto-added when WithSubAgents is set)
//
// This method is used by the tool filtering logic to distinguish user
// tools from framework tools.
func (a *LLMAgent) UserTools() []tool.Tool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	userTools := make([]tool.Tool, 0, len(a.tools))
	for _, t := range a.tools {
		name := t.Declaration().Name

		if a.userToolNames[name] {
			userTools = append(userTools, t)
			continue
		}
		if _, ok := t.(*itool.NamedTool); ok {
			userTools = append(userTools, t)
		}
	}

	return userTools
}

// FilterTools filters the list of tools based on the provided filter
// function.
func (a *LLMAgent) FilterTools(ctx context.Context) []tool.Tool {
	a.mu.RLock()
	tools := a.Tools()
	userToolNames := make(map[string]bool, len(a.userToolNames))
	for name, isUser := range a.userToolNames {
		userToolNames[name] = isUser
	}
	a.mu.RUnlock()

	filtered := make([]tool.Tool, 0, len(tools))
	filter := a.option.toolFilter
	for _, t := range tools {
		name := t.Declaration().Name
		isUser := userToolNames[name]
		if _, ok := t.(*itool.NamedTool); ok {
			isUser = true
		}

		if !isUser || filter == nil || filter(ctx, t) {
			filtered = append(filtered, t)
		}
	}

	return filtered
}

// CodeExecutor returns the code executor used by this agent.
// implements the agent.CodeExecutor interface.
// This allows the agent to execute code blocks in different environments.
func (a *LLMAgent) CodeExecutor() codeexecutor.CodeExecutor {
	return a.codeExecutor
}

// SetSubAgents replaces the sub-agents for this agent in a
// concurrency-safe way. This enables dynamic sub-agent discovery from
// registries without recreating the agent instance.
func (a *LLMAgent) SetSubAgents(subAgents []agent.Agent) {
	a.mu.Lock()
	a.subAgents = subAgents
	a.mu.Unlock()
}

// AddToolSet adds or replaces a tool set at runtime in a
// concurrency-safe way. If another ToolSet with the same Name()
// already exists, it will be replaced. Subsequent invocations see the
// updated tool list without recreating the agent.
func (a *LLMAgent) AddToolSet(toolSet tool.ToolSet) {
	if toolSet == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	name := toolSet.Name()
	replaced := false
	for i, ts := range a.option.ToolSets {
		if name != "" && ts.Name() == name {
			a.option.ToolSets[i] = toolSet
			replaced = true
			break
		}
	}
	if !replaced {
		a.option.ToolSets = append(a.option.ToolSets, toolSet)
	}
	a.registerTools()
}

// RemoveToolSet removes all tool sets whose Name() matches the given
// name. It returns true if at least one ToolSet was removed. Tools
// from the removed tool sets will no longer be exposed on future
// invocations.
func (a *LLMAgent) RemoveToolSet(name string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.option.ToolSets) == 0 {
		return false
	}

	dst := a.option.ToolSets[:0]
	removed := false
	for _, ts := range a.option.ToolSets {
		if ts.Name() == name {
			removed = true
			continue
		}
		dst = append(dst, ts)
	}
	if !removed {
		return false
	}
	a.option.ToolSets = dst
	a.registerTools()
	return true
}

// SetToolSets replaces the agent ToolSets with the provided slice in a
// concurrency-safe way. Subsequent invocations will see tools from
// exactly these ToolSets plus framework tools (knowledge, skills).
func (a *LLMAgent) SetToolSets(toolSets []tool.ToolSet) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(toolSets) == 0 {
		a.option.ToolSets = nil
	} else {
		copied := make([]tool.ToolSet, len(toolSets))
		copy(copied, toolSets)
		a.option.ToolSets = copied
	}
	a.registerTools()
}

// SetModel sets the model for this agent in a concurrency-safe way.
// This allows callers to manage multiple models externally and switch
// dynamically during runtime.
func (a *LLMAgent) SetModel(m model.Model) {
	a.mu.Lock()
	a.model = m
	a.mu.Unlock()
}

// SetModelByName switches the model by name in a concurrency-safe way.
// The model must be registered via WithModels option when creating the agent.
// Returns an error if the specified model name is not found.
func (a *LLMAgent) SetModelByName(modelName string) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	m, ok := a.models[modelName]
	if !ok {
		return fmt.Errorf("model %q not found in registered models", modelName)
	}

	a.model = m
	return nil
}

// SetInstruction updates the agent's instruction at runtime in a concurrency-safe way.
// Subsequent requests will use the new instruction without recreating the agent.
func (a *LLMAgent) SetInstruction(instruction string) {
	a.mu.Lock()
	a.instruction = instruction
	a.mu.Unlock()
}

// SetGlobalInstruction updates the agent's global system prompt at runtime.
// This affects the system-level prompt prepended to requests.
func (a *LLMAgent) SetGlobalInstruction(systemPrompt string) {
	a.mu.Lock()
	a.systemPrompt = systemPrompt
	a.mu.Unlock()
}

// getInstruction returns the current instruction with read lock.
func (a *LLMAgent) getInstruction() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.instruction
}

// getSystemPrompt returns the current system prompt with read lock.
func (a *LLMAgent) getSystemPrompt() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.systemPrompt
}
