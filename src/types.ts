/**
 * Type definitions for OpenAI-compatible API responses
 */

export interface OpenAIModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  max_model_len?: number;
  context_length?: number;
  max_context_length?: number;
}

export interface OpenAIModelsResponse {
  object: string;
  data: OpenAIModel[];
}

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
}

export interface OpenAIChatCompletionRequest {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
}

export interface OpenAIChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason: string | null;
  }>;
}

export interface OpenAIChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface GatewayConfig {
  serverUrl: string;
  apiKey?: string;
  requestTimeout: number;
  defaultMaxTokens: number;
  defaultMaxOutputTokens: number;
  enableToolCalling: boolean;
  parallelToolCalling: boolean;
  agentTemperature: number;
  toolExcludePatterns: string[];
  streamingIdleTimeout: number;
  maxRetries: number;
  retryDelay: number;
  systemPrompt: string;
  promptStripPatterns: string[];
  contextWarningThreshold: number;
  contextHardLimit: number;
  maxMessageHistory: number;
  enableProactiveTruncation: boolean;
  activePreset: string;
  showThinking: boolean;
  /** reasoning_budget override: null = use preset/server default, 0 = off, -1 = unlimited, >0 = token limit */
  reasoningBudget: number | null;
  /** Reasoning format: 'auto' = detect from template, 'deepseek' = force separate reasoning_content, 'none' = disable */
  reasoningFormat: 'auto' | 'deepseek' | 'none';
  /** Context usage % at which automatic LLM-based condensation is triggered (default 80) */
  contextCondensationThreshold: number;
}

/** Runtime stats tracked per-request for status bar display */
export interface RequestStats {
  lastTokensPerSec: number;
  lastContextPercent: number;
  lastInputTokens: number;
  lastOutputChars: number;
  requestCount: number;
}
