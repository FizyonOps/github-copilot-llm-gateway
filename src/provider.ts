import * as vscode from 'vscode';
import { GatewayClient } from './client';
import { GatewayConfig, OpenAIChatCompletionRequest } from './types';

/**
 * Language model provider for OpenAI-compatible inference servers
 */
export class GatewayProvider implements vscode.LanguageModelChatProvider {
  private readonly client: GatewayClient;
  private config: GatewayConfig;
  private readonly outputChannel: vscode.OutputChannel;
  // Store tool schemas for the current request to fill missing required properties
  private readonly currentToolSchemas: Map<string, unknown> = new Map();
  private statusBarItem: vscode.StatusBarItem | undefined;
  private readonly modelMetadata: Map<string, { maxTokens: number; maxOutputTokens: number }> = new Map();

  constructor(private readonly context: vscode.ExtensionContext, outputChannel?: vscode.OutputChannel) {
    this.outputChannel = outputChannel ?? vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
    this.config = this.loadConfig();
    this.client = new GatewayClient(this.config);

    // Watch for configuration changes
    context.subscriptions.push(
      vscode.workspace.onDidChangeConfiguration((e: vscode.ConfigurationChangeEvent) => {
        if (e.affectsConfiguration('github.copilot.llm-gateway')) {
          this.outputChannel.appendLine('Configuration changed, reloading...');
          try {
            this.reloadConfig();
          } catch (error) {
            this.outputChannel.appendLine(`ERROR: Failed to reload config: ${error}`);
            vscode.window.showErrorMessage('Failed to reload LLM Gateway configuration. Check settings.');
          }
        }
      })
    );
  }

  refreshModels(): void {
    this.modelMetadata.clear();
    this.outputChannel.appendLine('Model metadata cache cleared');
    vscode.window.showInformationMessage('GitHub Copilot LLM Gateway: Models refreshed');
  }

  /**
   * Map VS Code message role to OpenAI role string
   */
  private mapRole(role: vscode.LanguageModelChatMessageRole): string {
    if (role === vscode.LanguageModelChatMessageRole.User) {
      return 'user';
    }
    if (role === vscode.LanguageModelChatMessageRole.Assistant) {
      return 'assistant';
    }
    return 'user';
  }

  /**
   * Convert a tool result part to OpenAI format
   */
  private convertToolResultPart(part: vscode.LanguageModelToolResultPart): Record<string, unknown> {
    return {
      tool_call_id: part.callId,
      role: 'tool',
      content: typeof part.content === 'string' ? part.content : JSON.stringify(part.content),
    };
  }

  /**
   * Convert a tool call part to OpenAI format
   */
  private convertToolCallPart(part: vscode.LanguageModelToolCallPart): Record<string, unknown> {
    return {
      id: part.callId,
      type: 'function',
      function: {
        name: part.name,
        arguments: JSON.stringify(part.input),
      },
    };
  }

  // Helper method: convertMessages (kept for potential future use)
  private convertMessages(messages: readonly vscode.LanguageModelChatMessage[]): Record<string, unknown>[] {
    const openAIMessages: Record<string, unknown>[] = [];

    for (const msg of messages) {
      const role = this.mapRole(msg.role);
      const toolResults: Record<string, unknown>[] = [];
      const toolCalls: Record<string, unknown>[] = [];
      let textContent = '';

      for (const part of msg.content) {
        if (part instanceof vscode.LanguageModelTextPart) {
          textContent += part.value;
        } else if (part instanceof vscode.LanguageModelToolResultPart) {
          toolResults.push(this.convertToolResultPart(part));
        } else if (part instanceof vscode.LanguageModelToolCallPart) {
          toolCalls.push(this.convertToolCallPart(part));
        }
      }

      if (toolCalls.length > 0) {
        openAIMessages.push({ role: 'assistant', content: textContent || null, tool_calls: toolCalls });
      } else if (toolResults.length > 0) {
        openAIMessages.push(...toolResults);
      } else if (textContent) {
        openAIMessages.push({ role, content: textContent });
      }
    }

    return openAIMessages;
  }

  // Helper method: buildRequestOptions
  private buildRequestOptions(
    model: vscode.LanguageModelChatInformation,
    openAIMessages: any[],
    estimatedInputTokens: number
  ): any {
    const modelMaxContext = this.config.defaultMaxTokens || 32768;
    const bufferTokens = 128;
    let safeMaxOutputTokens = Math.min(
      this.config.defaultMaxOutputTokens || 2048,
      Math.floor(modelMaxContext - estimatedInputTokens - bufferTokens)
    );
    if (safeMaxOutputTokens < 64) {
      safeMaxOutputTokens = Math.max(64, Math.floor((this.config.defaultMaxOutputTokens || 2048) / 2));
    }

    this.outputChannel.appendLine(
      `Token estimate: input=${estimatedInputTokens}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
    );

    const requestOptions: any = {
      model: model.id,
      messages: openAIMessages,
      max_tokens: safeMaxOutputTokens,
      temperature: 0.7,
    };

    return requestOptions;
  }

  // Helper method: addTooling
  private addTooling(
    requestOptions: any,
    options: vscode.ProvideLanguageModelChatResponseOptions
  ): void {
    if (this.config.enableToolCalling && options.tools && options.tools.length > 0) {
      const validTools = this.filterValidTools(options.tools);

      if (validTools.length > 0) {
        requestOptions.tools = validTools.map((tool) => ({
          type: 'function',
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.inputSchema,
          },
        }));

        if (options.toolMode !== undefined) {
          requestOptions.tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
        }

        requestOptions.parallel_tool_calls = this.config.parallelToolCalling;
        this.outputChannel.appendLine(`Sending ${requestOptions.tools.length} valid tools to model (parallel: ${this.config.parallelToolCalling})`);
        if (options.tools.length > validTools.length) {
          this.outputChannel.appendLine(`Filtered out ${options.tools.length - validTools.length} invalid/disabled tools`);
        }
      }
    }
  }

  /**
   * Filter out invalid, disabled, or unsupported tools
   * VS Code may pass tools that don't exist or aren't properly configured
   */
  private filterValidTools(tools: readonly (vscode.LanguageModelToolInformation | vscode.LanguageModelChatTool)[]): (vscode.LanguageModelToolInformation | vscode.LanguageModelChatTool)[] {
    const validTools: (vscode.LanguageModelToolInformation | vscode.LanguageModelChatTool)[] = [];

    for (const tool of tools) {
      // Skip tools without a name
      if (!tool.name || tool.name.trim() === '') {
        this.outputChannel.appendLine(`  FILTERED: Tool without name`);
        continue;
      }

      // Skip tools with missing or invalid input schema
      if (!tool.inputSchema) {
        this.outputChannel.appendLine(`  FILTERED: ${tool.name} - missing input schema`);
        continue;
      }

      // Skip tools with invalid JSON schema (common with broken MCP tools)
      try {
        const schemaStr = JSON.stringify(tool.inputSchema);
        JSON.parse(schemaStr);
      } catch (e) {
        this.outputChannel.appendLine(`  FILTERED: ${tool.name} - invalid JSON schema: ${e}`);
        continue;
      }

      let excluded = false;
      for (const pattern of this.config.toolExcludePatterns) {
        try {
          if (new RegExp(pattern, 'i').test(tool.name)) {
            this.outputChannel.appendLine(`  FILTERED: ${tool.name} - matched exclude pattern: ${pattern}`);
            excluded = true;
            break;
          }
        } catch (e) {
          this.outputChannel.appendLine(`WARNING: Invalid toolExcludePatterns regex "${pattern}": ${e}`);
        }
      }
      if (excluded) {
        continue;
      }

      validTools.push(tool);
    }

    return validTools;
  }

  /**
   * Get default value for a JSON schema type
   */
  private getDefaultForType(schema: Record<string, unknown> | null | undefined): unknown {
    if (!schema?.type) {
      return null;
    }

    switch (schema.type) {
      case 'string':
        return schema.default ?? '';
      case 'number':
      case 'integer':
        return schema.default ?? 0;
      case 'boolean':
        return schema.default ?? false;
      case 'array':
        return schema.default ?? [];
      case 'object':
        return schema.default ?? {};
      case 'null':
        return null;
      default:
        // Handle union types like ["string", "null"]
        if (Array.isArray(schema.type)) {
          if (schema.type.includes('null')) {
            return null;
          }
          // Use first non-null type
          for (const t of schema.type) {
            if (t !== 'null') {
              return this.getDefaultForType({ ...schema, type: t });
            }
          }
        }
        return null;
    }
  }

  /**
   * Fill in missing required properties with default values based on the tool schema
   */
  private fillMissingRequiredProperties(args: Record<string, unknown>, toolName: string, toolSchema: Record<string, unknown> | null | undefined): Record<string, unknown> {
    if (!toolSchema?.required || !Array.isArray(toolSchema.required)) {
      return args;
    }

    const properties = (toolSchema.properties || {}) as Record<string, Record<string, unknown>>;
    const filledArgs = { ...args };
    const filledProperties: string[] = [];

    for (const requiredProp of toolSchema.required as string[]) {
      if (!(requiredProp in filledArgs)) {
        const propSchema = properties[requiredProp];
        const defaultValue = this.getDefaultForType(propSchema);
        filledArgs[requiredProp] = defaultValue;
        filledProperties.push(`${requiredProp}=${JSON.stringify(defaultValue)}`);
      }
    }

    if (filledProperties.length > 0) {
      this.outputChannel.appendLine(`  AUTO-FILLED missing required properties: ${filledProperties.join(', ')}`);
    }

    return filledArgs;
  }

  /**
   * Estimate token count for a message
   */
  private estimateMessageTokens(message: any): number {
    let text = '';
    if (typeof message.content === 'string') {
      text = message.content;
    } else if (message.content) {
      text = JSON.stringify(message.content);
    }
    if (message.tool_calls) {
      text += JSON.stringify(message.tool_calls);
    }
    // Conservative estimate: ~3.3 characters per token (closer to real tokenizers)
    return Math.ceil(text.length / 3.3);
  }

  /**
   * Truncate messages to fit within a token limit.
   * Strategy: Keep the first message (usually system prompt) and the most recent messages.
   * Remove older messages from the middle of the conversation.
   */
  private truncateMessagesToFit(messages: any[], maxTokens: number): any[] {
    if (messages.length === 0) {
      return messages;
    }

    // Calculate total tokens
    let totalTokens = 0;
    const messageTokens: number[] = [];
    for (const msg of messages) {
      const tokens = this.estimateMessageTokens(msg);
      messageTokens.push(tokens);
      totalTokens += tokens;
    }

    // If we're within limits, return as-is
    if (totalTokens <= maxTokens) {
      return messages;
    }

    this.outputChannel.appendLine(`Context overflow: ${totalTokens} tokens > ${maxTokens} limit. Truncating...`);

    // Strategy: Keep first message (system) and as many recent messages as possible
    const result: any[] = [];
    let usedTokens = 0;

    // Always keep the first message if it exists (usually system prompt)
    if (messages.length > 0) {
      if (messageTokens[0] <= maxTokens) {
        result.push(messages[0]);
        usedTokens += messageTokens[0];
      } else {
        this.outputChannel.appendLine(`WARNING: First message exceeds token limit (${messageTokens[0]} > ${maxTokens}). Including anyway.`);
        result.push(messages[0]);
        usedTokens += messageTokens[0];
      }
    }

    // Work backwards from the end, adding messages until we hit the limit
    const recentMessages: any[] = [];
    for (let i = messages.length - 1; i > 0; i--) {
      const msgTokens = messageTokens[i];
      if (usedTokens + msgTokens <= maxTokens) {
        recentMessages.unshift(messages[i]);
        usedTokens += msgTokens;
      } else {
        // Stop when we can't fit more messages
        break;
      }
    }

    // Combine first message with recent messages
    result.push(...recentMessages);

    this.outputChannel.appendLine(`Truncated: kept ${result.length}/${messages.length} messages, ~${usedTokens} tokens`);

    return result;
  }

  /**
   * Count occurrences of a character in a string
   */
  private countChar(str: string, char: string): number {
    // Escape regex special characters in the search char
    const escapePattern = /[.*+?^${}()|[\]\\]/g;
    const escapedChar = char.replaceAll(escapePattern, String.raw`\$&`);
    const regex = new RegExp(escapedChar, 'g');
    let count = 0;
    while (regex.exec(str) !== null) {
      count++;
    }
    return count;
  }

  /**
   * Balance unclosed braces/brackets in a JSON string
   */
  private balanceBrackets(str: string): string {
    let result = str;
    const missingBrackets = this.countChar(result, '[') - this.countChar(result, ']');
    const missingBraces = this.countChar(result, '{') - this.countChar(result, '}');

    if (Math.max(missingBrackets, missingBraces) > 50) {
      this.outputChannel.appendLine(`WARNING: JSON nesting depth exceeds 50, skipping bracket balancing`);
      return str;
    }

    result += ']'.repeat(Math.max(0, missingBrackets));
    result += '}'.repeat(Math.max(0, missingBraces));

    return result;
  }

  /**
   * Attempt to repair truncated or malformed JSON arguments
   */
  private tryRepairJson(jsonStr: string): unknown {
    if (!jsonStr || jsonStr.trim() === '') {
      return {};
    }

    // First, try direct parse
    try {
      return JSON.parse(jsonStr);
    } catch {
      // Continue to repair attempts
    }

    // Attempt repairs for common issues
    let repaired = jsonStr.trim();

    // Fix missing closing brackets/braces
    repaired = this.balanceBrackets(repaired);

    // Fix trailing comma before closing brace/bracket
    repaired = repaired.replaceAll(/,\s*([}\]])/g, '$1');

    // Fix truncated string value - close the string if odd number of quotes
    if (this.countChar(repaired, '"') % 2 !== 0) {
      repaired += '"';
      repaired = this.balanceBrackets(repaired);
    }

    try {
      const result = JSON.parse(repaired);
      if (repaired !== jsonStr) {
        this.outputChannel.appendLine(`JSON repaired successfully`);
      }
      return result;
    } catch {
      this.outputChannel.appendLine(`JSON repair failed. Original length: ${jsonStr.length}`);
      this.outputChannel.appendLine(`Original: ${jsonStr.substring(0, 500)}${jsonStr.length > 500 ? '...' : ''}`);
      return null;
    }
  }

  /**
   * Estimate token overhead for tool results in conversation history
   * Tool results often contain large file contents, search results, terminal outputs
   */
  private estimateToolResultOverhead(messages: any[]): number {
    let overhead = 0;

    for (const msg of messages) {
      if (msg.role === 'tool' && msg.content) {
        // Tool results are often large - estimate generously
        const contentLength = typeof msg.content === 'string'
          ? msg.content.length
          : JSON.stringify(msg.content).length;

        // Tool results have higher token density (~2.8 chars/token due to structured data)
        overhead += Math.ceil(contentLength / 2.8);
      }

      // Tool calls in assistant messages
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        overhead += msg.tool_calls.length * 50; // ~50 tokens per tool call metadata
      }
    }

    return overhead;
  }

  /**
   * Predict whether a request will exceed context limits BEFORE sending
   * This allows proactive truncation instead of waiting for timeout/failure
   */
  private predictContextOverflow(
    messages: any[],
    tools: readonly any[],
    modelId: string
  ): { willOverflow: boolean; estimatedTotalTokens: number; contextLimit: number; warningLevel: 'none' | 'warning' | 'critical'; recommendedAction: string; messageCount: number; toolOverhead: number } {
    const metadata = this.modelMetadata.get(modelId);
    const modelMaxContext = metadata?.maxTokens || this.config.defaultMaxTokens || 32768;

    // Step 1: Estimate input message tokens
    let messageTokens = 0;
    for (const msg of messages) {
      messageTokens += this.estimateMessageTokens(msg);
    }

    // Step 2: Estimate tool schema overhead
    const toolsSchema = tools.map((tool: any) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema
      }
    }));
    const toolsOverhead = Math.ceil(JSON.stringify(toolsSchema).length / 4);

    // Step 3: Estimate tool result overhead from history
    const toolResultOverhead = this.estimateToolResultOverhead(messages);

    // Step 4: Calculate total expected tokens (add 10% safety margin for estimation error)
    const totalEstimatedTokens = Math.ceil((messageTokens + toolsOverhead + toolResultOverhead) * 1.1);

    // Step 5: Determine warning level
    const warningThreshold = this.config.contextWarningThreshold || 80;
    const hardLimit = this.config.contextHardLimit || 95;

    const usagePercent = (totalEstimatedTokens / modelMaxContext) * 100;
    let warningLevel: 'none' | 'warning' | 'critical' = 'none';
    let willOverflow = false;
    let recommendedAction = 'None - within safe limits';

    if (usagePercent >= hardLimit) {
      warningLevel = 'critical';
      willOverflow = true;
      recommendedAction = 'Aggressive truncation required - remove oldest messages and large tool results';
    } else if (usagePercent >= warningThreshold) {
      warningLevel = 'warning';
      recommendedAction = 'Consider truncation - keep recent context only';
    }

    this.outputChannel.appendLine(`Context prediction: ${totalEstimatedTokens} tokens / ${modelMaxContext} limit (${usagePercent.toFixed(1)}%)`);
    this.outputChannel.appendLine(`  Message tokens: ${messageTokens}`);
    this.outputChannel.appendLine(`  Tools overhead: ${toolsOverhead}`);
    this.outputChannel.appendLine(`  Tool results: ${toolResultOverhead}`);
    this.outputChannel.appendLine(`  Warning level: ${warningLevel}`);

    return {
      willOverflow,
      estimatedTotalTokens: totalEstimatedTokens,
      contextLimit: modelMaxContext,
      warningLevel,
      recommendedAction,
      messageCount: messages.length,
      toolOverhead: toolsOverhead + toolResultOverhead
    };
  }

  /**
   * Apply smart truncation strategy based on overflow prediction
   * Prioritizes keeping: system prompt, recent user messages, important tool results
   */
  private applySmartTruncation(
    messages: any[],
    prediction: { willOverflow: boolean; estimatedTotalTokens: number; contextLimit: number; warningLevel: 'none' | 'warning' | 'critical'; recommendedAction: string; messageCount: number; toolOverhead: number },
    modelId: string
  ): any[] {
    const metadata = this.modelMetadata.get(modelId);
    const modelMaxContext = metadata?.maxTokens || this.config.defaultMaxTokens || 32768;
    const maxMessages = this.config.maxMessageHistory || 50;

    this.outputChannel.appendLine(`Applying smart truncation strategy`);
    this.outputChannel.appendLine(`  Target: fit within ${modelMaxContext} tokens`);
    this.outputChannel.appendLine(`  Current: ${prediction.estimatedTotalTokens} tokens across ${messages.length} messages`);

    // Strategy 1: Enforce message count limit
    let truncated = [...messages];
    if (truncated.length > maxMessages) {
      this.outputChannel.appendLine(`  Step 1: Enforce message count limit (${maxMessages} max)`);

      // Keep first message (system) + most recent messages
      const firstMsg = truncated[0];
      const recentMsgs = truncated.slice(-maxMessages + 1);
      truncated = [firstMsg, ...recentMsgs];
    }

    // Strategy 2: Remove large tool results from middle of history
    if (prediction.estimatedTotalTokens > modelMaxContext * 0.9) {
      this.outputChannel.appendLine(`  Step 2: Removing large tool results from older messages`);

      const result = [];
      let inMiddleRegion = false;

      for (let i = 0; i < truncated.length; i++) {
        const isSystem = i === 0;
        const isRecent = i >= truncated.length - 10;
        const isMiddle = !isSystem && !isRecent;

        if (isMiddle && truncated[i].role === 'tool') {
          // Skip tool results in middle region to save tokens
          const contentLen = typeof truncated[i].content === 'string'
            ? truncated[i].content.length
            : JSON.stringify(truncated[i].content).length;

          if (contentLen > 1000) {
            this.outputChannel.appendLine(`    Skipping large tool result (${contentLen} chars)`);
            continue;
          }
        }

        result.push(truncated[i]);
      }

      truncated = result;
    }

    // Strategy 3: Compress content in older messages
    if (prediction.estimatedTotalTokens > modelMaxContext * 0.85) {
      this.outputChannel.appendLine(`  Step 3: Compressing content in older messages`);

      for (let i = 1; i < truncated.length - 5; i++) {
        if (truncated[i].content && typeof truncated[i].content === 'string') {
          const originalLen = truncated[i].content.length;
          if (originalLen > 5000) {
            // Summarize long content - keep first and last 500 chars
            truncated[i].content = truncated[i].content.substring(0, 500) +
              '\n...[content compressed by LLM Gateway]...\n' +
              truncated[i].content.substring(originalLen - 500);
            this.outputChannel.appendLine(`    Compressed message ${i}: ${originalLen} → ${truncated[i].content.length} chars`);
          }
        }
      }
    }

    // Recalculate after truncation
    const newTokenEstimate = truncated.reduce((sum, msg) => sum + this.estimateMessageTokens(msg), 0);
    this.outputChannel.appendLine(`  Result: ${truncated.length} messages, ~${newTokenEstimate} tokens`);

    return truncated;
  }

  /**
   * Provide language model information - fetches available models from inference server
   */
  async provideLanguageModelChatInformation(
    options: { silent: boolean; },
    token: vscode.CancellationToken
  ): Promise<vscode.LanguageModelChatInformation[]> {
    try {
      this.outputChannel.appendLine('Fetching models from inference server...');
      const response = await this.client.fetchModels();

      const models: vscode.LanguageModelChatInformation[] = [];

      for (const model of response.data) {
        let maxTokens = this.config.defaultMaxTokens;
        let maxOutputTokens = this.config.defaultMaxOutputTokens;

        let serverContextSize = model.max_model_len || model.context_length || model.max_context_length;
        if (!serverContextSize || serverContextSize <= 0) {
          const details = await this.client.fetchModelDetails(model.id);
          if (details) {
            serverContextSize = details.max_model_len || details.context_length || details.max_context_length;
          }
        }

        if (serverContextSize && serverContextSize > 0) {
          maxTokens = serverContextSize;
          this.outputChannel.appendLine(`  ${model.id}: server_context=${serverContextSize}, using=${maxTokens}`);
        }
        maxOutputTokens = Math.min(this.config.defaultMaxOutputTokens, Math.floor(maxTokens / 2));

        this.modelMetadata.set(model.id, { maxTokens, maxOutputTokens });

        const modelInfo: vscode.LanguageModelChatInformation = {
          id: model.id,
          name: model.id,
          family: 'llm-gateway',
          maxInputTokens: maxTokens,
          maxOutputTokens: maxOutputTokens,
          version: '1.0.0',
          capabilities: {
            toolCalling: this.config.enableToolCalling
          },
        };

        models.push(modelInfo);
      }

      this.outputChannel.appendLine(`Found ${models.length} models: ${models.map(m => m.id).join(', ')}`);
      return models;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.outputChannel.appendLine(`ERROR: Failed to fetch models: ${errorMessage}`); if (!options.silent) {
        vscode.window.showErrorMessage(
          `GitHub Copilot LLM Gateway: Failed to fetch models. ${errorMessage}`,
          'Open Settings'
        ).then((selection: string | undefined) => {
          if (selection === 'Open Settings') {
            vscode.commands.executeCommand('workbench.action.openSettings', 'github.copilot.llm-gateway');
          }
        });
      }

      return [];
    }
  }

  /**
   * Process a message part using duck-typing for older VS Code versions
   */
  private processPartDuckTyped(
    part: unknown,
    toolResults: Record<string, unknown>[],
    toolCalls: Record<string, unknown>[]
  ): void {
    const anyPart = part as Record<string, unknown>;
    if ('callId' in anyPart && 'content' in anyPart && !('name' in anyPart)) {
      this.outputChannel.appendLine(`  Found tool result (duck-typed): callId=${anyPart.callId}`);
      toolResults.push({
        tool_call_id: anyPart.callId,
        role: 'tool',
        content: typeof anyPart.content === 'string' ? anyPart.content : JSON.stringify(anyPart.content),
      });
    } else if ('callId' in anyPart && 'name' in anyPart && 'input' in anyPart) {
      this.outputChannel.appendLine(`  Found tool call (duck-typed): callId=${anyPart.callId}, name=${anyPart.name}`);
      toolCalls.push({
        id: anyPart.callId,
        type: 'function',
        function: { name: anyPart.name, arguments: JSON.stringify(anyPart.input) },
      });
    }
  }

  /**
   * Convert a single VS Code message to OpenAI format with logging
   */
  private convertSingleMessageWithLogging(msg: vscode.LanguageModelChatMessage): Record<string, unknown>[] {
    const role = this.mapRole(msg.role);
    const toolResults: Record<string, unknown>[] = [];
    const toolCalls: Record<string, unknown>[] = [];
    let textContent = '';

    for (const part of msg.content) {
      if (part instanceof vscode.LanguageModelTextPart) {
        textContent += part.value;
      } else if (part instanceof vscode.LanguageModelToolResultPart) {
        this.outputChannel.appendLine(`  Found tool result: callId=${part.callId}`);
        toolResults.push(this.convertToolResultPart(part));
      } else if (part instanceof vscode.LanguageModelToolCallPart) {
        this.outputChannel.appendLine(`  Found tool call: callId=${part.callId}, name=${part.name}`);
        toolCalls.push(this.convertToolCallPart(part));
      } else {
        this.processPartDuckTyped(part, toolResults, toolCalls);
      }
    }

    const result: Record<string, unknown>[] = [];
    if (toolCalls.length > 0) {
      result.push({ role: 'assistant', content: textContent || null, tool_calls: toolCalls });
    } else if (toolResults.length > 0) {
      result.push(...toolResults);
    } else if (textContent) {
      result.push({ role, content: textContent });
    }
    return result;
  }

  /**
   * Calculate safe max output tokens based on input estimate
   */
  private calculateSafeMaxOutputTokens(modelId: string, estimatedInputTokens: number, toolsOverhead: number): number {
    const metadata = this.modelMetadata.get(modelId);
    const modelMaxContext = metadata?.maxTokens || this.config.defaultMaxTokens || 32768;
    const modelMaxOutput = metadata?.maxOutputTokens || this.config.defaultMaxOutputTokens || 2048;

    const totalEstimatedTokens = estimatedInputTokens + toolsOverhead;
    const conservativeInputEstimate = Math.ceil(totalEstimatedTokens * 1.2);
    const bufferTokens = 256;

    let safeMaxOutputTokens = Math.min(
      modelMaxOutput,
      Math.floor(modelMaxContext - conservativeInputEstimate - bufferTokens)
    );

    return Math.max(64, safeMaxOutputTokens);
  }

  /**
   * Build tools configuration for request
   */
  private buildToolsConfig(options: vscode.ProvideLanguageModelChatResponseOptions): Record<string, unknown>[] | undefined {
    if (!this.config.enableToolCalling || !options.tools || options.tools.length === 0) {
      return undefined;
    }

    this.currentToolSchemas.clear();

    const validTools = this.filterValidTools(options.tools);

    if (validTools.length === 0) {
      return undefined;
    }

    return validTools.map((tool) => {
      this.outputChannel.appendLine(`Tool: ${tool.name}`);
      this.outputChannel.appendLine(`  Description: ${tool.description?.substring(0, 100) || 'none'}...`);

      const schema = tool.inputSchema as Record<string, unknown> | undefined;
      this.currentToolSchemas.set(tool.name, schema);

      if (schema?.required && Array.isArray(schema.required)) {
        this.outputChannel.appendLine(`  Required properties: ${(schema.required as string[]).join(', ')}`);
      }

      return {
        type: 'function',
        function: { name: tool.name, description: tool.description, parameters: tool.inputSchema },
      };
    });
  }

  /**
   * Process a single tool call from the stream
   */
  private processToolCall(
    toolCall: { id: string; name: string; arguments: string },
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): void {
    this.outputChannel.appendLine(`\n=== TOOL CALL RECEIVED ===`);
    this.outputChannel.appendLine(`  ID: ${toolCall.id}`);
    this.outputChannel.appendLine(`  Name: ${toolCall.name}`);
    this.outputChannel.appendLine(`  Raw arguments: ${toolCall.arguments.substring(0, 1000)}${toolCall.arguments.length > 1000 ? '...' : ''}`);

    let args = this.tryRepairJson(toolCall.arguments) as Record<string, unknown> | null;

    if (args === null) {
      this.outputChannel.appendLine(`  ERROR: Failed to parse tool call arguments`);
      this.outputChannel.appendLine(`  Full arguments: ${toolCall.arguments}`);
      args = {};
    } else {
      const argKeys = Object.keys(args);
      this.outputChannel.appendLine(`  Parsed argument keys: ${argKeys.length > 0 ? argKeys.join(', ') : '(none)'}`);
    }

    const toolSchema = this.currentToolSchemas.get(toolCall.name) as Record<string, unknown> | undefined;
    if (toolSchema) {
      args = this.fillMissingRequiredProperties(args, toolCall.name, toolSchema);
    }

    this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
    progress.report(new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.name, args));
  }

  /**
   * Handle empty response from model
   */
  private async handleEmptyResponse(
    model: vscode.LanguageModelChatInformation,
    inputText: string,
    messageCount: number,
    toolCount: number,
    token: vscode.CancellationToken,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>
  ): Promise<void> {
    const inputTokenCount = await this.provideTokenCount(model, inputText, token);
    const modelMaxContext = this.config.defaultMaxTokens || 32768;

    this.outputChannel.appendLine(`WARNING: Model returned empty response with no tool calls.`);
    this.outputChannel.appendLine(`  Input tokens estimated: ${inputTokenCount}`);
    this.outputChannel.appendLine(`  Messages in conversation: ${messageCount}`);
    this.outputChannel.appendLine(`  Tools provided: ${toolCount}`);

    const errorHint = toolCount > 0
      ? `The model returned an empty response. This typically indicates the model failed to generate valid output with tool calling enabled. Check the inference server logs for errors.`
      : `The model returned an empty response. Check the inference server logs for details.`;

    this.outputChannel.appendLine(`  Issue: ${errorHint}`);

    const errorMessage = `I was unable to generate a response. ${errorHint}\n\n` +
      `Diagnostic info:\n- Model: ${model.id}\n- Tools provided: ${toolCount}\n` +
      `- Estimated input tokens: ${inputTokenCount}\n- Context limit: ${modelMaxContext}\n\n` +
      `Check the "GitHub Copilot LLM Gateway" output panel for detailed logs.`;

    progress.report(new vscode.LanguageModelTextPart(errorMessage));
  }

  /**
   * Handle chat request error
   */
  private handleChatError(error: unknown): never {
    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : '';

    this.outputChannel.appendLine(`ERROR: Chat request failed: ${errorMessage}`);
    if (errorStack) {
      this.outputChannel.appendLine(`Stack trace: ${errorStack}`);
    }

    const isToolError = errorMessage.includes('HarmonyError') || errorMessage.includes('unexpected tokens');

    if (isToolError) {
      this.outputChannel.appendLine('HINT: This appears to be a tool calling format error.');
      this.outputChannel.appendLine('The model may not support function calling properly.');
      this.outputChannel.appendLine('Try: 1) Using a different model, 2) Disabling tool calling in settings, or 3) Checking inference server logs');

      vscode.window.showErrorMessage(
        `GitHub Copilot LLM Gateway: Model failed to generate valid tool calls. This model may not support function calling. Check Output panel for details.`,
        'Open Output', 'Disable Tool Calling'
      ).then((selection: string | undefined) => {
        if (selection === 'Open Output') {
          this.outputChannel.show();
        } else if (selection === 'Disable Tool Calling') {
          vscode.workspace.getConfiguration('github.copilot.llm-gateway').update('enableToolCalling', false, vscode.ConfigurationTarget.Global);
        }
      });
    } else {
      vscode.window.showErrorMessage(`GitHub Copilot LLM Gateway: Chat request failed. ${errorMessage}`);
    }

    throw error;
  }

  /**
   * Provide language model chat response - streams responses from inference server
   */
  async provideLanguageModelChatResponse(
    model: vscode.LanguageModelChatInformation,
    messages: readonly vscode.LanguageModelChatMessage[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token: vscode.CancellationToken
  ): Promise<void> {
    this.outputChannel.appendLine(`Sending chat request to model: ${model.id}`);
    this.outputChannel.appendLine(`Tool mode: ${options.toolMode}, Tools: ${options.tools?.length || 0}`);
    this.outputChannel.appendLine(`Message count: ${messages.length}`);

    this.showWelcomeNotification(model.id);

    // Convert messages
    const openAIMessages: Record<string, unknown>[] = [];
    for (const msg of messages) {
      openAIMessages.push(...this.convertSingleMessageWithLogging(msg));
    }
    this.outputChannel.appendLine(`Converted to ${openAIMessages.length} OpenAI messages`);

    // Log message structure
    for (let i = 0; i < openAIMessages.length; i++) {
      const msg = openAIMessages[i];
      const toolCallId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : 'none';
      this.outputChannel.appendLine(`  Message ${i + 1}: role=${msg.role}, hasContent=${!!msg.content}, hasToolCalls=${!!msg.tool_calls}, toolCallId=${toolCallId}`);
    }

    // Predict context overflow BEFORE sending request
    const overflowPrediction = this.predictContextOverflow(
      openAIMessages,
      options.tools || [],
      model.id
    );

    if (overflowPrediction.warningLevel === 'critical') {
      this.outputChannel.appendLine(`⚠️ CRITICAL: Context overflow predicted before sending request`);
      this.outputChannel.appendLine(`  Estimated total: ${overflowPrediction.estimatedTotalTokens} tokens`);
      this.outputChannel.appendLine(`  Limit: ${overflowPrediction.contextLimit} tokens`);
      this.outputChannel.appendLine(`  Action: ${overflowPrediction.recommendedAction}`);

      // Show user warning
      vscode.window.showWarningMessage(
        `LLM Gateway: Request may exceed context limit (${overflowPrediction.estimatedTotalTokens}/${overflowPrediction.contextLimit} tokens). Gateway will truncate automatically.`,
        'View Details'
      ).then((selection) => {
        if (selection === 'View Details') {
          this.outputChannel.show();
        }
      });
    }

    // Apply proactive truncation if enabled
    let messagesToSend = openAIMessages;
    if (overflowPrediction.willOverflow && this.config.enableProactiveTruncation !== false) {
      this.outputChannel.appendLine(`Applying proactive truncation: ${overflowPrediction.recommendedAction}`);
      messagesToSend = this.applySmartTruncation(openAIMessages, overflowPrediction, model.id);
    }

    const truncatedMessages = messagesToSend;
    if (truncatedMessages.length < openAIMessages.length) {
      this.outputChannel.appendLine(`WARNING: Truncated conversation from ${openAIMessages.length} to ${truncatedMessages.length} messages to fit context limit`);
    }

    // Build input text for token estimation
    const inputText = truncatedMessages
      .map((m) => {
        let text = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
        if (m.tool_calls) { text += JSON.stringify(m.tool_calls); }
        return text;
      })
      .join('\n');

    const toolsOverhead = options.tools ? Math.ceil(JSON.stringify(options.tools).length / 4) : 0;
    const estimatedInputTokens = await this.provideTokenCount(model, inputText, token);
    const safeMaxOutputTokens = this.calculateSafeMaxOutputTokens(model.id, estimatedInputTokens, toolsOverhead);
    const modelMaxContext = overflowPrediction.contextLimit;

    this.outputChannel.appendLine(
      `Token estimate: input=${estimatedInputTokens}, tools=${toolsOverhead}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
    );

    // Build request with dynamic temperature
    const hasTools = this.config.enableToolCalling && options.tools && options.tools.length > 0;
    let temperature = 0.7;
    if (hasTools) {
      temperature = this.config.agentTemperature ?? 0;
    } else if (options.modelOptions && typeof options.modelOptions.temperature === 'number') {
      temperature = options.modelOptions.temperature;
    }

    const requestOptions: Record<string, unknown> = {
      model: model.id,
      messages: truncatedMessages,
      max_tokens: safeMaxOutputTokens,
      temperature,
    };

    const toolsConfig = this.buildToolsConfig(options);
    if (toolsConfig) {
      requestOptions.tools = toolsConfig;
      if (options.toolMode !== undefined) {
        requestOptions.tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
      }
      requestOptions.parallel_tool_calls = this.config.parallelToolCalling;
      this.outputChannel.appendLine(`Sending ${toolsConfig.length} tools to model (parallel: ${this.config.parallelToolCalling})`);
    }

    if (options.modelOptions) {
      const allowedKeys = ['temperature', 'top_p', 'top_k', 'frequency_penalty', 'presence_penalty', 'stop', 'seed'];
      for (const key of allowedKeys) {
        if (key in options.modelOptions && (options.modelOptions as Record<string, unknown>)[key] !== undefined) {
          (requestOptions as Record<string, unknown>)[key] = (options.modelOptions as Record<string, unknown>)[key];
        }
      }
    }

    // Log request (sanitized)
    const sanitizedRequest = { ...requestOptions };
    if (sanitizedRequest.messages && Array.isArray(sanitizedRequest.messages)) {
      sanitizedRequest.messages = sanitizedRequest.messages.map((msg: any) => {
        if (typeof msg === 'object' && msg !== null) {
          const sanitized = { ...msg };
          if (sanitized.content && typeof sanitized.content === 'string' && sanitized.content.length > 200) {
            sanitized.content = sanitized.content.substring(0, 200) + '...[truncated]';
          }
          return sanitized;
        }
        return msg;
      });
    }
    const debugRequest = JSON.stringify(sanitizedRequest, null, 2);
    this.outputChannel.appendLine(debugRequest.length > 2000 ? `Request (truncated): ${debugRequest.substring(0, 2000)}...` : `Request: ${debugRequest}`);

    await this.executeStreamWithRetry(requestOptions, model, openAIMessages, options, token, progress, inputText);
  }

  private async executeStreamWithRetry(
    requestOptions: Record<string, unknown>,
    model: vscode.LanguageModelChatInformation,
    openAIMessages: Record<string, unknown>[],
    options: vscode.ProvideLanguageModelChatResponseOptions,
    token: vscode.CancellationToken,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    inputText: string,
    retryAttempt: number = 0
  ): Promise<void> {
    const maxContextRetries = 2;

    try {
      let totalContent = '';
      let totalToolCalls = 0;
      const idleTimeout = this.config.streamingIdleTimeout;
      const iterator = this.client.streamChatCompletion(
        requestOptions as unknown as OpenAIChatCompletionRequest,
        token,
        this.config.maxRetries,
        this.config.retryDelay
      )[Symbol.asyncIterator]();

      while (true) {
        if (token.isCancellationRequested) { break; }

        let result: IteratorResult<any>;
        if (idleTimeout > 0) {
          let timeoutId: ReturnType<typeof setTimeout> | undefined;
          const timeoutPromise = new Promise<never>((_, reject) => {
            timeoutId = setTimeout(() => reject(new Error('Streaming idle timeout')), idleTimeout);
          });
          try {
            result = await Promise.race([iterator.next(), timeoutPromise]);
          } catch (e: any) {
            if (e.message === 'Streaming idle timeout') {
              this.outputChannel.appendLine(`WARNING: Streaming idle timeout after ${idleTimeout}ms`);
              break;
            }
            throw e;
          } finally {
            if (timeoutId !== undefined) { clearTimeout(timeoutId); }
          }
        } else {
          result = await iterator.next();
        }

        if (result.done) { break; }
        const chunk = result.value;

        if (chunk.content) {
          totalContent += chunk.content;
          progress.report(new vscode.LanguageModelTextPart(chunk.content));
        }

        if (chunk.finished_tool_calls?.length) {
          for (const toolCall of chunk.finished_tool_calls) {
            totalToolCalls++;
            this.processToolCall(toolCall, progress);
          }
        }
      }

      this.outputChannel.appendLine(`Completed chat request, received ${totalContent.length} characters, ${totalToolCalls} tool calls`);

      if (totalContent.length === 0 && totalToolCalls === 0) {
        await this.handleEmptyResponse(model, inputText, openAIMessages.length, requestOptions.tools ? (requestOptions.tools as unknown[]).length : 0, token, progress);
      }
    } catch (error: any) {
      if (error.isContextOverflow && retryAttempt < maxContextRetries) {
        const serverPromptTokens = error.promptTokens;
        const serverContextSize = error.contextSize;
        this.outputChannel.appendLine(`Context overflow from server: ${serverPromptTokens} prompt tokens > ${serverContextSize} context. Retry ${retryAttempt + 1}/${maxContextRetries}`);

        if (serverContextSize && serverContextSize > 0) {
          this.modelMetadata.set(model.id, {
            maxTokens: serverContextSize,
            maxOutputTokens: Math.min(this.config.defaultMaxOutputTokens, Math.floor(serverContextSize / 2)),
          });
        }

        const currentMessages = requestOptions.messages as any[];
        const targetTokens = (serverContextSize || this.config.defaultMaxTokens) * 0.7;
        const aggressivelyTruncated = this.truncateMessagesToFit(currentMessages, targetTokens);

        this.outputChannel.appendLine(`Aggressive truncation: ${currentMessages.length} -> ${aggressivelyTruncated.length} messages, target ${targetTokens} tokens`);

        const newInputText = aggressivelyTruncated
          .map((m: any) => {
            let text = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
            if (m.tool_calls) { text += JSON.stringify(m.tool_calls); }
            return text;
          })
          .join('\n');
        const toolsOverhead = requestOptions.tools ? Math.ceil(JSON.stringify(requestOptions.tools).length / 3.3) : 0;
        const newEstimate = await this.provideTokenCount(model, newInputText, token);
        const newMaxOutput = this.calculateSafeMaxOutputTokens(model.id, newEstimate, toolsOverhead);

        const retryOptions = {
          ...requestOptions,
          messages: aggressivelyTruncated,
          max_tokens: newMaxOutput,
        };

        return this.executeStreamWithRetry(retryOptions, model, openAIMessages, options, token, progress, newInputText, retryAttempt + 1);
      }

      this.handleChatError(error);
    }
  }

  /**
   * Provide token count estimation
   */
  async provideTokenCount(
    model: vscode.LanguageModelChatInformation,
    text: string | vscode.LanguageModelChatMessage,
    token: vscode.CancellationToken
  ): Promise<number> {
    if (token.isCancellationRequested) {
      return 0;
    }
    // Conservative approximation: ~3.3 characters per token
    // This is closer to real tokenizer behavior than the common 4 chars/token estimate
    let content: string;

    if (typeof text === 'string') {
      content = text;
    } else {
      // Filter and extract only text parts from the message content
      content = text.content
        .filter((part): part is vscode.LanguageModelTextPart => part instanceof vscode.LanguageModelTextPart)
        .map((part) => part.value)
        .join('');
    }

    const estimatedTokens = Math.ceil(content.length / 3.3);
    return estimatedTokens;
  }

  private showWelcomeNotification(modelId: string): void {
    if (!this.statusBarItem) {
      this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
      this.statusBarItem.command = 'workbench.action.openSettings';
      this.context.subscriptions.push(this.statusBarItem);
    }
    this.statusBarItem.text = `$(hubot) LLM: ${modelId}`;
    this.statusBarItem.tooltip = 'GitHub Copilot LLM Gateway - Click to open settings';
    this.statusBarItem.show();
  }

  /**
   * Load configuration from VS Code settings
   */
  private loadConfig(): GatewayConfig {
    const config = vscode.workspace.getConfiguration('github.copilot.llm-gateway');

    const cfg: GatewayConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      apiKey: config.get<string>('apiKey', ''),
      requestTimeout: config.get<number>('requestTimeout', 300000),
      defaultMaxTokens: config.get<number>('defaultMaxTokens', 32768),
      defaultMaxOutputTokens: config.get<number>('defaultMaxOutputTokens', 4096),
      enableToolCalling: config.get<boolean>('enableToolCalling', true),
      parallelToolCalling: config.get<boolean>('parallelToolCalling', true),
      agentTemperature: config.get<number>('agentTemperature', 0),
      toolExcludePatterns: config.get<string[]>('toolExcludePatterns', ['^search_view_results$', '^view_results$', '^vscode_internal', '^extension_', '^unknown_']),
      streamingIdleTimeout: config.get<number>('streamingIdleTimeout', 120000),
      maxRetries: config.get<number>('maxRetries', 2),
      retryDelay: config.get<number>('retryDelay', 1000),
      contextWarningThreshold: config.get<number>('contextWarningThreshold', 75),
      contextHardLimit: config.get<number>('contextHardLimit', 85),
      maxMessageHistory: config.get<number>('maxMessageHistory', 50),
      enableProactiveTruncation: config.get<boolean>('enableProactiveTruncation', true),
    };

    // Validate requestTimeout
    if (cfg.requestTimeout <= 0) {
      this.outputChannel.appendLine(`ERROR: requestTimeout must be > 0; using default 60000`);
      cfg.requestTimeout = 60000;
    }

    try {
      new URL(cfg.serverUrl);
    } catch {
      this.outputChannel.appendLine(`ERROR: Invalid server URL: ${cfg.serverUrl}`);
      throw new Error(`Invalid server URL: ${cfg.serverUrl}`);
    }

    // Validate defaultMaxOutputTokens relative to defaultMaxTokens
    if (cfg.defaultMaxOutputTokens >= cfg.defaultMaxTokens) {
      const adjusted = Math.max(64, cfg.defaultMaxTokens - 256);
      this.outputChannel.appendLine(
        `WARNING: github.copilot.llm-gateway.defaultMaxOutputTokens (${cfg.defaultMaxOutputTokens}) >= defaultMaxTokens (${cfg.defaultMaxTokens}). Adjusting to ${adjusted}.`
      );
      vscode.window.showWarningMessage(
        `GitHub Copilot LLM Gateway: 'defaultMaxOutputTokens' was >= 'defaultMaxTokens'. Adjusted to ${adjusted} to avoid request errors.`
      );
      cfg.defaultMaxOutputTokens = adjusted;
    }

    return cfg;
  }

  private dispose(): void {
    this.statusBarItem?.dispose();
  }

  private reloadConfig(): void {
    this.config = this.loadConfig();
    this.client.updateConfig(this.config);
    this.modelMetadata.clear();
    this.outputChannel.appendLine('Configuration reloaded, model metadata cache cleared');
  }
}
