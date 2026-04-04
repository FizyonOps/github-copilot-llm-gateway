import * as vscode from 'vscode';
import { existsSync } from 'node:fs';
import * as path from 'node:path';
import { GatewayClient } from './client';
import { GatewayConfig, OpenAIChatCompletionRequest, RequestStats } from './types';
import { collectWorkspaceInstructionFiles } from './instructions';
import { BackendType, BUILTIN_PRESETS, SamplingPreset, detectBackend, filterParamsForBackend, getAllowedKeysForBackend, resolvePresetsDir, ensurePresetsDir, loadPresetsFromDir, getPresetFilePath, detectTemplateFromJinja, TemplateCaps } from './presets';

const DEFAULT_PROMPT_STRIP_PATTERNS = [
  '(?:You are (?:an expert )?AI programming assistant,\\s*)?working with a user in the VS Code editor\\.',
  'When asked for your name, you must respond with (?:\\\\?")?GitHub Copilot(?:\\\\?")?\\.',
  'When asked about the model you are using, you must state that you are using [\\s\\S]*?(?:\\\\n|\\r?\\n|$)',
  "Follow the user's requirements carefully\\s*&\\s*to the letter\\.",
  'Follow Microsoft content policies\\.',
  'Avoid content that violates copyrights\\.',
  `If you are asked to generate content that is harmful, hateful, racist, sexist, lewd, or violent, only respond with (?:\\\\?")?Sorry, I can't assist with that(?:\\\\?")?\\.?`,
];

/**
 * Model variant suffix encoding: "<base-id>__rb:<budget>"
 * budget: 0 = off, -1 = unlimited, >0 = token cap
 */
const MODEL_VARIANT_RE = /__rb:(-?\d+)$/;

function parseModelVariant(modelId: string): { realId: string; variantBudget: number | null } {
  const m = MODEL_VARIANT_RE.exec(modelId);
  if (!m) return { realId: modelId, variantBudget: null };
  return { realId: modelId.slice(0, -m[0].length), variantBudget: parseInt(m[1], 10) };
}

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
  private presetStatusBarItem: vscode.StatusBarItem | undefined;
  private readonly modelMetadata: Map<string, { maxTokens: number; maxOutputTokens: number }> = new Map();
  private lastKnownModels: vscode.LanguageModelChatInformation[] = [];
  private detectedBackend: BackendType = 'unknown';
  private detectedTemplate: TemplateCaps = { template: 'unknown', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
  private presetsDir: string = '';
  private loadedPresets: Record<string, SamplingPreset> = {};
  private presetWatcher: vscode.FileSystemWatcher | undefined;
  private requestStats: RequestStats = { lastTokensPerSec: 0, lastContextPercent: 0, lastInputTokens: 0, lastOutputChars: 0, requestCount: 0 };
  /** Stored conversation for export (model id → messages) */
  private lastConversation: { modelId: string; messages: Array<{ role: string; content: string; thinking?: string }> } | undefined;

  constructor(private readonly context: vscode.ExtensionContext, outputChannel?: vscode.OutputChannel) {
    this.outputChannel = outputChannel ?? vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
    this.config = this.loadConfig();
    this.client = new GatewayClient(this.config);

    // Initialize file-based presets
    this.initializePresets();

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

  /** Public accessors for chat participant */
  getStats(): RequestStats { return this.requestStats; }
  getBackend(): BackendType { return this.detectedBackend; }
  getTemplate(): TemplateCaps { return this.detectedTemplate; }
  getConfig(): GatewayConfig { return this.config; }
  getPresets(): Record<string, SamplingPreset> { return this.loadedPresets; }
  getLastConversation() { return this.lastConversation; }

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

  private estimateToolsTokens(tools: readonly any[] | undefined): number {
    if (!tools || tools.length === 0) {
      return 0;
    }

    return Math.ceil(JSON.stringify(tools).length / 3.3);
  }

  private estimatePromptTokens(messages: any[], tools?: readonly any[]): number {
    return messages.reduce((sum, msg) => sum + this.estimateMessageTokens(msg), 0) + this.estimateToolsTokens(tools);
  }

  private truncateTextToTokenBudget(text: string, maxTokens: number, label: string): string {
    const maxChars = Math.max(256, Math.floor(maxTokens * 3.3));
    if (text.length <= maxChars) {
      return text;
    }

    const marker = `\n...[${label} truncated by LLM Gateway]...\n`;
    const availableChars = Math.max(128, maxChars - marker.length);
    const headChars = Math.max(64, Math.floor(availableChars / 2));
    const tailChars = Math.max(64, availableChars - headChars);

    if (text.length <= headChars + tailChars + marker.length) {
      return text;
    }

    return text.substring(0, headChars) + marker + text.substring(text.length - tailChars);
  }

  private sanitizeVisibleContent(text: string): string {
    return text.replaceAll(/<\/?think>/gi, '');
  }

  private normalizeReasoningContent(text: string): string {
    return text
      .replaceAll(/<\/?think>/gi, '')
      .replaceAll(/<\/?details>/gi, '')
      .replaceAll(/<\/?summary>/gi, '')
      .trim();
  }

  private stripPromptPatterns(text: string, label: string): string {
    if (!text || this.config.promptStripPatterns.length === 0) {
      return text;
    }

    let sanitizedText = text;
    let removedMatches = 0;

    for (const pattern of this.config.promptStripPatterns) {
      try {
        const regex = new RegExp(pattern, 'giu');
        sanitizedText = sanitizedText.replace(regex, () => {
          removedMatches += 1;
          return '';
        });
      } catch (error) {
        this.outputChannel.appendLine(`WARNING: Invalid promptStripPatterns regex "${pattern}": ${error}`);
      }
    }

    if (removedMatches === 0) {
      return text;
    }

    sanitizedText = sanitizedText
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n[ \t]+/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/(?:\\n\s*){3,}/g, '\\n\\n')
      .trim();

    this.outputChannel.appendLine(`Stripped ${removedMatches} prompt boilerplate match(es) from ${label}`);

    return sanitizedText;
  }

  private sanitizePromptMessages(messages: Record<string, unknown>[]): Record<string, unknown>[] {
    const sanitizedMessages: Record<string, unknown>[] = [];

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      if (message.role === 'tool' || typeof message.content !== 'string') {
        sanitizedMessages.push(message);
        continue;
      }

      const role = typeof message.role === 'string' ? message.role : 'unknown';
      const sanitizedContent = this.stripPromptPatterns(message.content, `message ${i + 1} (${role})`);

      if (sanitizedContent === message.content) {
        sanitizedMessages.push(message);
        continue;
      }

      if (sanitizedContent === '') {
        if (Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
          sanitizedMessages.push({ ...message, content: null });
        } else {
          this.outputChannel.appendLine(`Dropped empty ${role} message ${i + 1} after prompt pattern stripping`);
        }
        continue;
      }

      sanitizedMessages.push({ ...message, content: sanitizedContent });
    }

    return sanitizedMessages;
  }

  private async buildWorkspaceInstructionsPrompt(
    conversationMessages: readonly Record<string, unknown>[]
  ): Promise<string | undefined> {
    const instructionFiles = await collectWorkspaceInstructionFiles();
    if (instructionFiles.length === 0) {
      return undefined;
    }

    const existingContent = conversationMessages
      .map((message) => typeof message.content === 'string' ? message.content : '')
      .filter((content) => content !== '')
      .join('\n\n');

    const sections: string[] = [];
    for (const file of instructionFiles) {
      if (existingContent.includes(file.content)) {
        this.outputChannel.appendLine(`Skipping workspace instructions from ${file.label}; already present in chat context`);
        continue;
      }

      this.outputChannel.appendLine(`Loaded workspace instructions from ${file.label} (${file.content.length} chars)`);
      sections.push(`[Workspace instructions source: ${file.label}]\n${file.content}`);
    }

    if (sections.length === 0) {
      return undefined;
    }

    return sections.join('\n\n---\n\n');
  }

  private async buildCombinedSystemPrompt(
    conversationMessages: readonly Record<string, unknown>[]
  ): Promise<string | undefined> {
    const sections: string[] = [];
    const configuredSystemPrompt = this.config.systemPrompt.trim();
    if (configuredSystemPrompt) {
      sections.push(configuredSystemPrompt);
    }

    const workspaceInstructionsPrompt = await this.buildWorkspaceInstructionsPrompt(conversationMessages);
    if (workspaceInstructionsPrompt) {
      sections.push(workspaceInstructionsPrompt);
    }

    const workspacePathGuardPrompt = this.buildWorkspacePathGuardPrompt();
    if (workspacePathGuardPrompt) {
      sections.push(workspacePathGuardPrompt);
    }

    if (sections.length === 0) {
      return undefined;
    }

    return sections.join('\n\n');
  }

  private getWorkspaceRoots(): string[] {
    return (vscode.workspace.workspaceFolders ?? [])
      .map((folder) => path.normalize(folder.uri.fsPath))
      .filter((root) => root.length > 0);
  }

  private isWithinWorkspaceRoots(candidatePath: string, workspaceRoots: readonly string[]): boolean {
    const normalizedCandidate = path.normalize(candidatePath);
    for (const root of workspaceRoots) {
      if (normalizedCandidate === root || normalizedCandidate.startsWith(root + path.sep)) {
        return true;
      }
    }
    return false;
  }

  private normalizePathArgument(rawPath: string): string {
    if (rawPath.startsWith('file://')) {
      try {
        return path.normalize(vscode.Uri.parse(rawPath).fsPath);
      } catch {
        return path.normalize(rawPath);
      }
    }

    return path.normalize(rawPath);
  }

  private tryRemapToWorkspacePath(sourcePath: string, workspaceRoots: readonly string[]): string | undefined {
    const normalizedSource = path.normalize(sourcePath);
    const pathParts = normalizedSource.split(path.sep).filter(Boolean);

    if (pathParts.length < 2) {
      return undefined;
    }

    const srcIndex = pathParts.indexOf('src');
    if (srcIndex >= 0 && srcIndex < pathParts.length - 1) {
      const sourceSuffix = pathParts.slice(srcIndex);
      for (const root of workspaceRoots) {
        const candidate = path.join(root, ...sourceSuffix);
        if (existsSync(candidate)) {
          return candidate;
        }
      }
    }

    const maxSuffixDepth = Math.min(pathParts.length, 8);
    for (let suffixDepth = maxSuffixDepth; suffixDepth >= 2; suffixDepth--) {
      const sourceSuffix = pathParts.slice(-suffixDepth);
      for (const root of workspaceRoots) {
        const candidate = path.join(root, ...sourceSuffix);
        if (existsSync(candidate)) {
          return candidate;
        }
      }
    }

    return undefined;
  }

  private sanitizeWorkspacePathArgument(
    toolName: string,
    argumentName: string,
    value: string,
    workspaceRoots: readonly string[]
  ): string | undefined {
    const trimmedValue = value.trim();
    if (!trimmedValue || /^https?:\/\//i.test(trimmedValue) || /^untitled:/i.test(trimmedValue)) {
      return value;
    }

    const normalizedPath = this.normalizePathArgument(trimmedValue);
    if (!path.isAbsolute(normalizedPath)) {
      return value;
    }

    if (this.isWithinWorkspaceRoots(normalizedPath, workspaceRoots)) {
      return normalizedPath;
    }

    const remappedPath = this.tryRemapToWorkspacePath(normalizedPath, workspaceRoots);
    if (remappedPath) {
      this.outputChannel.appendLine(`  Remapped ${toolName}.${argumentName}: ${normalizedPath} -> ${remappedPath}`);
      return remappedPath;
    }

    // Enhanced logging for security diagnostics
    const pathType = path.isAbsolute(trimmedValue) ? 'absolute' : 'relative';
    this.outputChannel.appendLine(`⚠️ SECURITY WARNING: ${toolName}.${argumentName} points outside workspace roots`);
    this.outputChannel.appendLine(`  Path type: ${pathType}`);
    this.outputChannel.appendLine(`  Actual path: ${normalizedPath}`);
    this.outputChannel.appendLine(`  Workspace roots being compared:`);
    for (const root of workspaceRoots) {
      this.outputChannel.appendLine(`    - ${root}`);
    }
    this.outputChannel.appendLine(`  → Blocking tool call until the model provides a workspace-local path`);
    return undefined;
  }

  private sanitizeToolPathArguments(toolName: string, args: Record<string, unknown>): {
    args: Record<string, unknown>;
    blockedPathArguments: string[];
  } {
    const normalizedToolName = toolName.toLowerCase();
    if (
      normalizedToolName.includes('memory')
      || normalizedToolName.includes('gemini')
      || normalizedToolName.includes('chrome')
      || normalizedToolName.includes('browser')
      || normalizedToolName.includes('network')
      || normalizedToolName.includes('bookmark')
      || normalizedToolName.includes('history')
    ) {
      return { args, blockedPathArguments: [] };
    }

    const workspaceRoots = this.getWorkspaceRoots();
    if (workspaceRoots.length === 0) {
      return { args, blockedPathArguments: [] };
    }

    const sanitizedArgs: Record<string, unknown> = { ...args };
    const blockedPathArguments: string[] = [];
    const pathArgumentKeys = ['filePath', 'path', 'dirPath', 'workspaceFolder', 'includePattern'];

    for (const key of pathArgumentKeys) {
      const rawValue = sanitizedArgs[key];
      if (typeof rawValue === 'string') {
        const sanitizedValue = this.sanitizeWorkspacePathArgument(toolName, key, rawValue, workspaceRoots);
        if (sanitizedValue === undefined) {
          blockedPathArguments.push(key);
          delete sanitizedArgs[key];
        } else {
          sanitizedArgs[key] = sanitizedValue;
        }
      }
    }

    const filePathsValue = sanitizedArgs.filePaths;
    if (Array.isArray(filePathsValue)) {
      const safeFilePaths: unknown[] = [];
      filePathsValue.forEach((entry, index) => {
        if (typeof entry !== 'string') {
          safeFilePaths.push(entry);
          return;
        }
        const sanitizedValue = this.sanitizeWorkspacePathArgument(toolName, `filePaths[${index}]`, entry, workspaceRoots);
        if (sanitizedValue === undefined) {
          blockedPathArguments.push(`filePaths[${index}]`);
          return;
        }
        safeFilePaths.push(sanitizedValue);
      });
      sanitizedArgs.filePaths = safeFilePaths;
    }

    return { args: sanitizedArgs, blockedPathArguments };
  }

  private coerceTaggedParameterValue(rawValue: string): unknown {
    const trimmed = rawValue.trim();
    if (trimmed === '') {
      return '';
    }

    if (/^-?\d+$/.test(trimmed)) {
      const parsedInt = Number.parseInt(trimmed, 10);
      if (Number.isFinite(parsedInt)) {
        return parsedInt;
      }
    }

    if (/^-?\d*\.\d+$/.test(trimmed)) {
      const parsedFloat = Number.parseFloat(trimmed);
      if (Number.isFinite(parsedFloat)) {
        return parsedFloat;
      }
    }

    if (/^true$/i.test(trimmed)) {
      return true;
    }

    if (/^false$/i.test(trimmed)) {
      return false;
    }

    if (/^null$/i.test(trimmed)) {
      return null;
    }

    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        return JSON.parse(trimmed);
      } catch {
        return trimmed;
      }
    }

    return trimmed;
  }

  private extractTaggedToolCalls(text: string): {
    remainingText: string;
    toolCalls: Array<{ id: string; name: string; arguments: string }>;
  } {
    if (!text) {
      return { remainingText: text, toolCalls: [] };
    }

    const toolCalls: Array<{ id: string; name: string; arguments: string }> = [];
    const toolCallRegex = /<tool_call>\s*([\s\S]*?)<\/tool_call>/gi;
    const callIdSeed = Date.now().toString(36);
    let match: RegExpExecArray | null;

    while ((match = toolCallRegex.exec(text)) !== null) {
      const toolBlock = match[1] || '';
      const functionMatch = /<function=([^\s>]+)>\s*([\s\S]*?)<\/function>/i.exec(toolBlock);
      if (!functionMatch) {
        continue;
      }

      const functionName = functionMatch[1].trim();
      if (!functionName) {
        continue;
      }

      const paramsBlock = functionMatch[2] || '';
      const params: Record<string, unknown> = {};
      const parameterRegex = /<parameter=([^\s>]+)>\s*([\s\S]*?)<\/parameter>/gi;
      let parameterMatch: RegExpExecArray | null;

      while ((parameterMatch = parameterRegex.exec(paramsBlock)) !== null) {
        const paramName = parameterMatch[1].trim();
        if (!paramName) {
          continue;
        }
        params[paramName] = this.coerceTaggedParameterValue(parameterMatch[2] || '');
      }

      toolCalls.push({
        id: `tagged_${callIdSeed}_${toolCalls.length}`,
        name: functionName,
        arguments: JSON.stringify(params),
      });
    }

    if (toolCalls.length === 0) {
      return { remainingText: text, toolCalls };
    }

    const remainingText = text
      .replace(/<tool_call>\s*[\s\S]*?<\/tool_call>/gi, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return { remainingText, toolCalls };
  }

  private isPureTaggedToolCallPayload(text: string): boolean {
    if (!text.trim()) {
      return false;
    }

    const stripped = text.replace(/<tool_call>\s*[\s\S]*?<\/tool_call>/gi, '').trim();
    return stripped.length === 0;
  }

  private containsTranscriptLikeTaggedToolCallMarkers(text: string): boolean {
    const markers = [
      '<|im_start|>',
      '<|im_end|>',
      '<tool_response>',
      '"role":',
      '"tool_calls"',
      '[{"$mid"',
      '```',
    ];

    return markers.some((marker) => text.includes(marker));
  }

  private getRecoverableTaggedToolCalls(
    text: string,
    sourceLabel: string
  ): {
    remainingText: string;
    toolCalls: Array<{ id: string; name: string; arguments: string }>;
  } | undefined {
    if (!text.trim() || !text.includes('<tool_call>')) {
      return undefined;
    }

    const extracted = this.extractTaggedToolCalls(text);
    if (extracted.toolCalls.length === 0) {
      return undefined;
    }

    if (this.isPureTaggedToolCallPayload(text)) {
      return extracted;
    }

    if (this.containsTranscriptLikeTaggedToolCallMarkers(text)) {
      this.outputChannel.appendLine(
        `Skipped tagged tool-call fallback for ${sourceLabel}: response looks like echoed transcript content.`
      );
      return undefined;
    }

    const remainingText = extracted.remainingText.trim();
    const lineCount = remainingText ? remainingText.split(/\r?\n/).length : 0;
    const isShortMixedPreamble = remainingText.length <= 1600 && lineCount <= 24;

    if (isShortMixedPreamble) {
      this.outputChannel.appendLine(
        `Recovering tagged tool calls from ${sourceLabel} with short mixed prose preamble (${remainingText.length} chars, ${lineCount} lines).`
      );
      return extracted;
    }

    this.outputChannel.appendLine(
      `Skipped tagged tool-call fallback for ${sourceLabel}: mixed content was too large to trust (${remainingText.length} chars, ${lineCount} lines).`
    );
    return undefined;
  }

  private buildWorkspacePathGuardPrompt(): string | undefined {
    const workspaceRoots = this.getWorkspaceRoots();
    if (workspaceRoots.length === 0) {
      return undefined;
    }

    const listedRoots = workspaceRoots.map((root) => `- ${root}`).join('\n');
    return [
      'Workspace path boundaries for tool calls:',
      'Use only files and folders inside the active workspace roots below.',
      listedRoots,
      'Prefer workspace-relative paths when a tool supports both absolute and relative paths.',
      'If an absolute path is outside these roots, do not use it directly in tool arguments.',
    ].join('\n');
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

    const tokenBudget = Math.max(128, Math.floor(maxTokens));

    // Calculate total tokens
    let totalTokens = 0;
    const messageTokens: number[] = [];
    for (const msg of messages) {
      const tokens = this.estimateMessageTokens(msg);
      messageTokens.push(tokens);
      totalTokens += tokens;
    }

    // If we're within limits, return as-is
    if (totalTokens <= tokenBudget) {
      return messages;
    }

    this.outputChannel.appendLine(`Context overflow: ${totalTokens} tokens > ${tokenBudget} limit. Truncating...`);

    // Strategy: Prefer the system prompt if present, then keep as many recent messages as possible.
    const result: any[] = [];
    let usedTokens = 0;
    let startIndex = 0;

    // Preserve a system prompt, but compress it if it would dominate the budget.
    if (messages[0]?.role === 'system') {
      startIndex = 1;

      let firstMessage = messages[0];
      let firstMessageTokens = messageTokens[0];
      const preferredSystemBudget = Math.max(128, Math.floor(tokenBudget * 0.2));

      if (typeof firstMessage.content === 'string' && firstMessageTokens > preferredSystemBudget) {
        firstMessage = {
          ...firstMessage,
          content: this.truncateTextToTokenBudget(firstMessage.content, preferredSystemBudget, 'system prompt'),
        };
        firstMessageTokens = this.estimateMessageTokens(firstMessage);
        this.outputChannel.appendLine(`Compressed system prompt: ${messageTokens[0]} -> ${firstMessageTokens} tokens`);
      }

      if (firstMessageTokens < tokenBudget) {
        result.push(firstMessage);
        usedTokens += firstMessageTokens;
      } else {
        this.outputChannel.appendLine(`WARNING: Dropping oversized system prompt after compression (${firstMessageTokens} tokens)`);
      }
    }

    // Work backwards from the end, adding messages until we hit the limit
    const recentMessages: any[] = [];
    for (let i = messages.length - 1; i >= startIndex; i--) {
      const remainingBudget = tokenBudget - usedTokens;
      if (remainingBudget <= 0) {
        break;
      }

      let candidate = messages[i];
      let candidateTokens = messageTokens[i];

      if (candidateTokens > remainingBudget) {
        if (typeof candidate.content === 'string' && remainingBudget >= 128) {
          const compressedCandidate = {
            ...candidate,
            content: this.truncateTextToTokenBudget(candidate.content, remainingBudget, `${candidate.role} message`),
          };
          const compressedTokens = this.estimateMessageTokens(compressedCandidate);

          if (compressedTokens <= remainingBudget && compressedTokens < candidateTokens) {
            candidate = compressedCandidate;
            candidateTokens = compressedTokens;
            this.outputChannel.appendLine(`Compressed ${candidate.role} message at index ${i}: ${messageTokens[i]} -> ${candidateTokens} tokens`);
          } else {
            continue;
          }
        } else {
          continue;
        }
      }

      recentMessages.unshift(candidate);
      usedTokens += candidateTokens;
    }

    // Combine first message with recent messages
    result.push(...recentMessages);

    if (result.length === 0) {
      const fallback = messages[messages.length - 1];
      if (typeof fallback?.content === 'string') {
        const compressedFallback = {
          ...fallback,
          content: this.truncateTextToTokenBudget(fallback.content, tokenBudget, `${fallback.role} message`),
        };
        result.push(compressedFallback);
        usedTokens = this.estimateMessageTokens(compressedFallback);
      } else if (fallback) {
        result.push(fallback);
        usedTokens = this.estimateMessageTokens(fallback);
      }
    }

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

    // Fix truncated string value - close the string if odd number of quotes
    if (this.countChar(repaired, '"') % 2 !== 0) {
      repaired += '"';
    }

    // Fix missing closing brackets/braces after repairing quotes so appended braces
    // do not get swallowed into an unterminated string value.
    repaired = this.balanceBrackets(repaired);

    // Fix trailing comma before closing brace/bracket
    repaired = repaired.replaceAll(/,\s*([}\]])/g, '$1');

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
    const toolsOverhead = this.estimateToolsTokens(toolsSchema);

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
    const warningThreshold = this.config.contextWarningThreshold || 75;
    const targetUsagePercent = prediction.warningLevel === 'critical'
      ? Math.max(55, warningThreshold - 15)
      : Math.max(60, warningThreshold - 5);
    const targetTokenBudget = Math.max(256, Math.floor(modelMaxContext * (targetUsagePercent / 100)));

    this.outputChannel.appendLine(`Applying smart truncation strategy`);
    this.outputChannel.appendLine(`  Target: fit within ${modelMaxContext} tokens`);
    this.outputChannel.appendLine(`  Current: ${prediction.estimatedTotalTokens} tokens across ${messages.length} messages`);
    this.outputChannel.appendLine(`  Proactive target budget: ${targetTokenBudget} tokens (${targetUsagePercent}% of context)`);

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
    if (prediction.warningLevel !== 'none') {
      this.outputChannel.appendLine(`  Step 2: Removing large tool results from older messages`);

      const result = [];
      const toolResultCharLimit = prediction.warningLevel === 'critical' ? 1000 : 4000;

      for (let i = 0; i < truncated.length; i++) {
        const isSystem = i === 0;
        const isRecent = i >= truncated.length - 10;
        const isMiddle = !isSystem && !isRecent;

        if (isMiddle && truncated[i].role === 'tool') {
          // Skip tool results in middle region to save tokens
          const contentLen = typeof truncated[i].content === 'string'
            ? truncated[i].content.length
            : JSON.stringify(truncated[i].content).length;

          if (contentLen > toolResultCharLimit) {
            this.outputChannel.appendLine(`    Skipping large tool result (${contentLen} chars)`);
            // Also remove the preceding assistant message with matching tool_calls to maintain pairing integrity
            const previousMessage = result.length > 0 ? result[result.length - 1] : undefined;
            const previousToolCalls = Array.isArray(previousMessage?.tool_calls) ? previousMessage.tool_calls as any[] : [];
            const toolCallId = typeof truncated[i].tool_call_id === 'string' ? truncated[i].tool_call_id : undefined;
            const hasMatchingToolCall = toolCallId
              ? previousToolCalls.some((toolCall: any) => toolCall?.id === toolCallId)
              : previousToolCalls.length > 0;
            if (previousMessage?.role === 'assistant' && hasMatchingToolCall) {
              result.pop();
              this.outputChannel.appendLine(`    Also removing paired assistant tool call message`);
            }
            continue;
          }
        }

        result.push(truncated[i]);
      }

      truncated = result;
    }

    // Strategy 3: Compress content in older messages
    if (prediction.warningLevel !== 'none') {
      this.outputChannel.appendLine(`  Step 3: Compressing content in older messages`);

      const compressThreshold = prediction.warningLevel === 'critical' ? 2000 : 3500;
      const compressTargetTokens = prediction.warningLevel === 'critical' ? 300 : 500;

      for (let i = 1; i < truncated.length - 5; i++) {
        if (truncated[i].content && typeof truncated[i].content === 'string') {
          const originalLen = truncated[i].content.length;
          if (originalLen > compressThreshold) {
            // Use object spread to create a new message object instead of mutating shared references
            truncated[i] = {
              ...truncated[i],
              content: this.truncateTextToTokenBudget(
                truncated[i].content,
                compressTargetTokens,
                'content'
              ),
            };
            this.outputChannel.appendLine(`    Compressed message ${i}: ${originalLen} → ${(truncated[i] as any).content.length} chars`);
          }
        }
      }
    }

    const estimatedTokensAfterCompaction = truncated.reduce((sum, msg) => sum + this.estimateMessageTokens(msg), 0);
    if (estimatedTokensAfterCompaction > targetTokenBudget) {
      this.outputChannel.appendLine(`  Step 4: Trimming to proactive budget`);
      truncated = this.truncateMessagesToFit(truncated, targetTokenBudget);
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

      // Detect backend type from response
      this.detectedBackend = detectBackend(response as any);
      this.outputChannel.appendLine(`Detected backend: ${this.detectedBackend}`);

      // For llama.cpp: fetch /props to get the actual Jinja chat template and detect capabilities
      if (this.detectedBackend === 'llamacpp') {
        const props = await this.client.fetchProps();
        if (props?.chat_template) {
          this.detectedTemplate = detectTemplateFromJinja(props.chat_template);
          // Override hasNativeThinking from the server's own detection
          if (props.chat_template_explicit_reasoning === true) {
            this.detectedTemplate.hasNativeThinking = true;
          }
          this.outputChannel.appendLine(
            `Chat template: ${this.detectedTemplate.template} (toolRole=${this.detectedTemplate.supportsToolRole}, systemRole=${this.detectedTemplate.supportsSystemRole}, nativeThinking=${this.detectedTemplate.hasNativeThinking})`
          );
        } else {
          this.outputChannel.appendLine('Could not fetch /props — template capabilities unknown, assuming full support');
        }
      } else {
        // Non-llamacpp: assume full support
        this.detectedTemplate = { template: 'unknown', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
      }

      this.updatePresetStatusBar();

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

        // Register reasoning-budget variants so users can select the thinking
        // level directly from the model picker (works on any backend).
        const THINKING_VARIANTS: Array<{ suffix: string; label: string }> = [
          { suffix: '__rb:0', label: '[think: off]' },
          { suffix: '__rb:1024', label: '[think: 1k]' },
          { suffix: '__rb:4096', label: '[think: 4k]' },
          { suffix: '__rb:16384', label: '[think: 16k]' },
          { suffix: '__rb:-1', label: '[think: ∞]' },
        ];
        for (const v of THINKING_VARIANTS) {
          const variantId = model.id + v.suffix;
          this.modelMetadata.set(variantId, { maxTokens, maxOutputTokens });
          models.push({
            id: variantId,
            name: `${model.id} ${v.label}`,
            family: 'llm-gateway',
            maxInputTokens: maxTokens,
            maxOutputTokens: maxOutputTokens,
            version: '1.0.0',
            capabilities: { toolCalling: this.config.enableToolCalling },
          });
        }
      }

      this.lastKnownModels = models.map(model => ({
        ...model,
        capabilities: { ...model.capabilities },
      }));
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

      if (this.lastKnownModels.length > 0) {
        this.outputChannel.appendLine(`WARNING: Returning ${this.lastKnownModels.length} cached models to avoid dropping the active selection during a transient failure.`);
        return this.lastKnownModels.map(model => ({
          ...model,
          capabilities: { ...model.capabilities },
        }));
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

  private hasMessageRole(messages: readonly Record<string, unknown>[], role: string): boolean {
    return messages.some((message) => typeof message.role === 'string' && message.role === role);
  }

  private findLastUserMessage(messages: readonly Record<string, unknown>[]): Record<string, unknown> | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (message.role === 'user' && typeof message.content === 'string' && message.content.trim() !== '') {
        return { ...message };
      }
    }

    return undefined;
  }

  private createSyntheticUserMessage(messages: readonly Record<string, unknown>[]): Record<string, unknown> {
    const hasToolContext = messages.some(
      (message) => message.role === 'tool' || (Array.isArray(message.tool_calls) && message.tool_calls.length > 0)
    );

    return {
      role: 'user',
      content: hasToolContext
        ? 'Continue the current conversation using the assistant and tool results provided.'
        : 'Continue the current conversation using the provided context.',
    };
  }

  private ensureConversationHasUserTurn(
    messages: Record<string, unknown>[],
    sourceMessages: readonly Record<string, unknown>[],
    tokenBudget?: number
  ): Record<string, unknown>[] {
    if (this.hasMessageRole(messages, 'user')) {
      return messages;
    }

    const restoredUserMessage = this.findLastUserMessage(sourceMessages);
    const requiredUserMessage = restoredUserMessage
      ?? this.createSyntheticUserMessage(sourceMessages.length > 0 ? sourceMessages : messages);
    const repairedMessages = [...messages];
    const insertIndex = repairedMessages[0]?.role === 'system' ? 1 : 0;
    let insertedMessage = { ...requiredUserMessage };
    let compressedInsertedMessage = false;

    if (typeof tokenBudget === 'number' && tokenBudget > 0 && typeof insertedMessage.content === 'string') {
      const reservedBudget = Math.max(64, Math.floor(tokenBudget * 0.15));
      if (this.estimateMessageTokens(insertedMessage) > reservedBudget) {
        insertedMessage = {
          ...insertedMessage,
          content: this.truncateTextToTokenBudget(insertedMessage.content, reservedBudget, 'user message'),
        };
        compressedInsertedMessage = true;
      }
    }

    repairedMessages.splice(insertIndex, 0, insertedMessage);

    if (typeof tokenBudget === 'number' && tokenBudget > 0) {
      while (this.estimatePromptTokens(repairedMessages) > tokenBudget && repairedMessages.length > insertIndex + 1) {
        const removableIndex = repairedMessages.findIndex(
          (message, index) => index > insertIndex && message.role !== 'user' && message.role !== 'system'
        );

        if (removableIndex === -1) {
          break;
        }

        const removedRole = repairedMessages[removableIndex].role;
        repairedMessages.splice(removableIndex, 1);
        this.outputChannel.appendLine(`Removed ${removedRole} message to preserve a required user turn within budget`);
      }
    }

    const insertionMode = compressedInsertedMessage ? 'restored/compressed' : 'restored';
    if (restoredUserMessage) {
      this.outputChannel.appendLine(`Recovered missing user turn in outbound request (${insertionMode})`);
    } else {
      this.outputChannel.appendLine(`Inserted synthetic user turn for tool-only/model-only conversation`);
    }

    return repairedMessages;
  }

  private buildInputText(messages: readonly Record<string, unknown>[]): string {
    return messages
      .map((message) => {
        let text = typeof message.content === 'string' ? message.content : JSON.stringify(message.content || '');
        if (message.tool_calls) {
          text += JSON.stringify(message.tool_calls);
        }
        return text;
      })
      .join('\n');
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
      this.outputChannel.appendLine(`  BLOCKED: Skipping malformed tool call`);
      this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
      progress.report(new vscode.LanguageModelTextPart(`Skipped malformed tool call '${toolCall.name}' because its arguments were truncated or invalid.`));
      return;
    } else {
      const argKeys = Object.keys(args);
      this.outputChannel.appendLine(`  Parsed argument keys: ${argKeys.length > 0 ? argKeys.join(', ') : '(none)'}`);
    }

    const toolSchema = this.currentToolSchemas.get(toolCall.name) as Record<string, unknown> | undefined;
    if (toolSchema) {
      args = this.fillMissingRequiredProperties(args, toolCall.name, toolSchema);
    }

    const sanitization = this.sanitizeToolPathArguments(toolCall.name, args);
    if (sanitization.blockedPathArguments.length > 0) {
      this.outputChannel.appendLine(`  BLOCKED: Path arguments outside workspace roots: ${sanitization.blockedPathArguments.join(', ')}`);
      this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
      progress.report(new vscode.LanguageModelTextPart(`Blocked tool call '${toolCall.name}' because it referenced files outside the active workspace.`));
      return;
    }

    args = sanitization.args;

    this.outputChannel.appendLine(`=== END TOOL CALL ===\n`);
    progress.report(new vscode.LanguageModelToolCallPart(toolCall.id, toolCall.name, args));
  }

  private async closeAndDrainStreamIterator(
    iterator: AsyncIterator<any>,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    allowToolExecution: boolean
  ): Promise<number> {
    if (typeof iterator.return !== 'function') {
      return 0;
    }

    let drainedToolCalls = 0;

    try {
      let closeResult = await iterator.return(undefined);
      while (!closeResult.done) {
        const chunk = closeResult.value;
        if (allowToolExecution && chunk?.finished_tool_calls?.length) {
          for (const toolCall of chunk.finished_tool_calls) {
            drainedToolCalls++;
            this.processToolCall(toolCall, progress);
          }
        }
        closeResult = await iterator.next();
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.outputChannel.appendLine(`DEBUG: Stream iterator cleanup ended with: ${errorMessage}`);
    }

    return drainedToolCalls;
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

  private isCancellationError(error: unknown, token?: vscode.CancellationToken): boolean {
    if (token?.isCancellationRequested) {
      return true;
    }

    if (error instanceof vscode.CancellationError) {
      return true;
    }

    const errorMessage = error instanceof Error ? error.message : String(error);
    return /request cancelled|request canceled|cancelled|canceled/i.test(errorMessage);
  }

  /**
   * Handle chat request error
   */
  private handleChatError(
    error: unknown,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    token?: vscode.CancellationToken
  ): void {
    if (this.isCancellationError(error, token)) {
      this.outputChannel.appendLine('Chat request cancelled');
      return;
    }

    const errorMessage = error instanceof Error ? error.message : String(error);
    const errorStack = error instanceof Error ? error.stack : '';

    this.outputChannel.appendLine(`ERROR: Chat request failed: ${errorMessage}`);
    if (errorStack) {
      this.outputChannel.appendLine(`Stack trace: ${errorStack}`);
    }

    const isToolError = errorMessage.includes('HarmonyError') || errorMessage.includes('unexpected tokens');
    const chatErrorMessage = `I couldn't complete the request with the selected model.\n\nReason: ${errorMessage}\n\nThe selected model remains active. Check the \"GitHub Copilot LLM Gateway\" output panel for details.`;
    progress.report(new vscode.LanguageModelTextPart(chatErrorMessage));

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
  }

  /**
   * Adapt outbound messages to match the detected chat template's capabilities.
   *
   * - Templates that don't support 'tool' role: merge tool results into the next user message
   *   as a bracketed block, and strip tool_calls from assistant messages.
   * - Templates that don't support 'system' role (e.g. Gemma): prepend the system content
   *   to the first user message.
   *
   * This prevents silent prompt corruption when llama.cpp applies the chat template
   * and encounters roles it doesn't know how to format.
   */
  private adaptMessagesForTemplate(messages: Record<string, unknown>[]): Record<string, unknown>[] {
    const caps = this.detectedTemplate;

    // Fast-path: nothing to adapt
    if (caps.supportsToolRole && caps.supportsSystemRole) {
      return messages;
    }

    const adapted: Record<string, unknown>[] = [];
    let pendingSystemContent = '';
    let pendingToolContent = '';

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      const role = typeof msg.role === 'string' ? msg.role : '';

      // ── System role ───────────────────────────────────────────────
      if (role === 'system' && !caps.supportsSystemRole) {
        const content = typeof msg.content === 'string' ? msg.content : '';
        pendingSystemContent += (pendingSystemContent ? '\n\n' : '') + content;
        this.outputChannel.appendLine(`Template adaptation: system message merged into next user message (${content.length} chars)`);
        continue;
      }

      // ── Tool role ─────────────────────────────────────────────────
      if (role === 'tool' && !caps.supportsToolRole) {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content ?? '');
        const callId = typeof msg.tool_call_id === 'string' ? msg.tool_call_id : '';
        pendingToolContent += `\n[Tool result${callId ? ` (${callId})` : ''}]:\n${content}`;
        this.outputChannel.appendLine(`Template adaptation: tool result queued for merging into next user message`);
        continue;
      }

      // ── User role — flush pending system/tool content ─────────────
      if (role === 'user') {
        let userContent = typeof msg.content === 'string' ? msg.content : '';
        let prefix = '';
        if (pendingSystemContent) {
          prefix += `[System context]:\n${pendingSystemContent}\n\n`;
          pendingSystemContent = '';
        }
        if (pendingToolContent) {
          prefix += pendingToolContent.trimStart() + '\n\n';
          pendingToolContent = '';
        }
        if (prefix) {
          userContent = prefix + userContent;
        }
        adapted.push({ ...msg, content: userContent });
        continue;
      }

      // ── Assistant role — strip tool_calls if template can't handle them ──
      if (role === 'assistant' && !caps.supportsToolRole && Array.isArray(msg.tool_calls)) {
        const { tool_calls: _tc, ...rest } = msg as any;
        // Convert tool calls into readable text appended to content
        const toolCallText = (msg.tool_calls as any[])
          .map((tc: any) => `[Called: ${tc?.function?.name ?? 'tool'}(${tc?.function?.arguments ?? ''})]`)
          .join('\n');
        const content = typeof msg.content === 'string' && msg.content
          ? `${msg.content}\n${toolCallText}`
          : toolCallText;
        adapted.push({ ...rest, content });
        continue;
      }

      adapted.push(msg);
    }

    // Flush any leftover pending content as a synthetic user message
    if (pendingToolContent) {
      adapted.push({ role: 'user', content: pendingToolContent.trim() });
      this.outputChannel.appendLine('Template adaptation: added synthetic user message for trailing tool results');
    }

    return adapted;
  }

  private getModelContextLimit(modelId: string): number {
    const { realId } = parseModelVariant(modelId);
    return this.modelMetadata.get(modelId)?.maxTokens
      || this.modelMetadata.get(realId)?.maxTokens
      || this.config.defaultMaxTokens
      || 32768;
  }

  private renderMessageForCondensation(msg: Record<string, unknown>): string | undefined {
    const role = typeof msg.role === 'string' ? msg.role : 'unknown';

    if (role === 'tool') {
      const content = typeof msg.content === 'string' ? msg.content : '';
      const excerpt = content.length > 500 ? `${content.substring(0, 500)}...[truncated]` : content;
      return excerpt.trim() ? `[tool result]: ${excerpt}` : undefined;
    }

    let content = typeof msg.content === 'string' ? msg.content : (msg.content ? JSON.stringify(msg.content) : '');
    if (content.length > 2000) {
      content = `${content.substring(0, 2000)}...[truncated]`;
    }
    if (!content.trim()) {
      return undefined;
    }

    const parts = [`[${role}]: ${content}`];
    const toolCalls = Array.isArray((msg as any).tool_calls) ? (msg as any).tool_calls as any[] : [];
    if (toolCalls.length > 0) {
      const names = toolCalls.map((tc: any) => tc?.function?.name || 'unknown').join(', ');
      parts.push(`[${role} called tools: ${names}]`);
    }

    return parts.join('\n');
  }

  private mergeSummaryIntoMessage(
    message: Record<string, unknown>,
    summaryBlock: string,
    prepend: boolean = false
  ): Record<string, unknown> {
    const existingContent = typeof message.content === 'string'
      ? message.content
      : (message.content ? JSON.stringify(message.content) : '');
    const mergedContent = existingContent.trim().length === 0
      ? summaryBlock
      : prepend
        ? `${summaryBlock}\n\n${existingContent}`
        : `${existingContent}\n\n---\n\n${summaryBlock}`;

    return {
      ...message,
      content: mergedContent,
    };
  }

  /**
   * Condense older conversation messages using the LLM to produce a compact summary.
   * Keeps the system prompt (first message) and the last KEEP_RECENT messages intact.
   * Everything in between is sent to the LLM for summarization.
   * Returns the original messages unchanged if condensation fails or is unnecessary.
   */
  private async condenseMessages(
    messages: Record<string, unknown>[],
    modelId: string,
    token: vscode.CancellationToken
  ): Promise<Record<string, unknown>[]> {
    const KEEP_RECENT = 6;
    const modelContextLimit = this.getModelContextLimit(modelId);
    const condensationInputBudgetTokens = Math.max(1024, Math.min(8192, Math.floor(modelContextLimit * 0.35)));
    const condensationOutputBudgetTokens = Math.max(512, Math.min(2048, Math.floor(modelContextLimit * 0.12)));
    const condensationInputBudgetChars = Math.max(512, Math.floor(condensationInputBudgetTokens * 3.3));

    if (messages.length <= KEEP_RECENT + 2) {
      return messages;
    }

    const hasSystem = messages[0]?.role === 'system';
    const startIdx = hasSystem ? 1 : 0;
    const recentStart = Math.max(startIdx, messages.length - KEEP_RECENT);
    const oldMessages = messages.slice(startIdx, recentStart);
    const recentMessages = messages.slice(recentStart);

    if (oldMessages.length === 0) {
      return messages;
    }

    const renderableOldMessages = oldMessages
      .map((msg, index) => {
        const text = this.renderMessageForCondensation(msg);
        return text ? { index, text } : undefined;
      })
      .filter((entry): entry is { index: number; text: string } => !!entry);

    const selectedMessages: Array<{ index: number; text: string }> = [];
    let usedChars = 0;
    for (let i = renderableOldMessages.length - 1; i >= 0; i--) {
      const candidate = renderableOldMessages[i];
      const separatorChars = selectedMessages.length > 0 ? 2 : 0;
      if (usedChars + separatorChars + candidate.text.length <= condensationInputBudgetChars) {
        selectedMessages.unshift(candidate);
        usedChars += separatorChars + candidate.text.length;
        continue;
      }

      if (selectedMessages.length === 0) {
        const truncatedText = this.truncateTextToTokenBudget(
          candidate.text,
          Math.max(128, Math.floor(condensationInputBudgetTokens * 0.5)),
          'condensation input'
        );
        selectedMessages.unshift({ index: candidate.index, text: truncatedText });
      }
      break;
    }

    const totalRenderableCount = renderableOldMessages.length;
    const summarizedCount = selectedMessages.length;
    const omittedCount = Math.max(0, totalRenderableCount - summarizedCount);
    const conversationText = selectedMessages.map((entry) => entry.text).join('\n\n');
    if (conversationText.trim().length < 100) {
      return messages;
    }

    const condensationPrompt =
      `Summarize this conversation concisely, preserving:\n` +
      `- Key technical decisions and requirements\n` +
      `- File paths, function/variable names, and code references\n` +
      `- Current work state (what is done, what is pending)\n` +
      `- Errors encountered and their resolutions\n` +
      `- Important constraints or user preferences\n\n` +
      (omittedCount > 0
        ? `Note: ${omittedCount} oldest older messages were omitted from this condensation input because the conversation already exceeded the summarization budget. Preserve continuity from the included newer history.\n\n`
        : '') +
      `Respond with ONLY the summary. No meta-commentary.\n\n---\n\n${conversationText}`;

    this.outputChannel.appendLine(
      `Condensation: summarizing ${summarizedCount}/${totalRenderableCount} older messages (${conversationText.length} chars, budget ~${condensationInputBudgetTokens} tokens)...`
    );

    try {
      let condensationRequestMessages: Record<string, unknown>[] = [
        { role: 'system', content: 'You are a conversation summarizer. Be concise but preserve all technically important details.' },
        { role: 'user', content: condensationPrompt },
      ];

      if (this.detectedBackend === 'llamacpp') {
        condensationRequestMessages = this.adaptMessagesForTemplate(condensationRequestMessages);
      }

      condensationRequestMessages = this.ensureConversationHasUserTurn(condensationRequestMessages, condensationRequestMessages);

      const summary = await this.client.chatComplete(
        {
          model: modelId,
          messages: condensationRequestMessages,
          max_tokens: condensationOutputBudgetTokens,
          temperature: 0.1,
        } as any,
        token
      );

      if (!summary || summary.trim().length === 0) {
        this.outputChannel.appendLine('Condensation returned empty result, falling back to truncation');
        return messages;
      }

      const condensed: Record<string, unknown>[] = [];
      const summaryHeader = omittedCount > 0
        ? `[Condensed conversation context — latest ${summarizedCount} of ${totalRenderableCount} older messages summarized]`
        : `[Condensed conversation context — ${totalRenderableCount} older messages summarized]`;
      const summaryBlock = `${summaryHeader}\n${summary.trim()}`;

      if (hasSystem) {
        condensed.push(this.mergeSummaryIntoMessage(messages[0], summaryBlock));
        condensed.push(...recentMessages);
      } else {
        const recentWithSummary = [...recentMessages];
        const firstUserIndex = recentWithSummary.findIndex((msg) => msg.role === 'user');
        if (firstUserIndex >= 0) {
          recentWithSummary[firstUserIndex] = this.mergeSummaryIntoMessage(recentWithSummary[firstUserIndex], summaryBlock, true);
        } else {
          recentWithSummary.unshift({ role: 'user', content: summaryBlock });
        }
        condensed.push(...recentWithSummary);
      }

      const oldTokens = this.estimatePromptTokens(messages);
      const newTokens = this.estimatePromptTokens(condensed);
      this.outputChannel.appendLine(
        `Condensation complete: ${messages.length} → ${condensed.length} messages, ~${oldTokens} → ~${newTokens} tokens (saved ~${oldTokens - newTokens})`
      );

      return condensed;
    } catch (error) {
      if (this.isCancellationError(error, token)) {
        throw error;
      }
      this.outputChannel.appendLine(`Condensation failed: ${error}. Falling back to truncation.`);
      return messages;
    }
  }

  private async condenseAndTrimIfNeeded(
    messages: Record<string, unknown>[],
    modelId: string,
    tools: readonly any[] | undefined,
    token: vscode.CancellationToken,
    logPrefix: string
  ): Promise<Record<string, unknown>[]> {
    let condensed = await this.condenseMessages(messages, parseModelVariant(modelId).realId, token);
    const postCondensePrediction = this.predictContextOverflow(condensed, tools || [], modelId);
    if (postCondensePrediction.warningLevel !== 'none' && this.config.enableProactiveTruncation !== false) {
      this.outputChannel.appendLine(
        `${logPrefix}: still at ${((postCondensePrediction.estimatedTotalTokens / postCondensePrediction.contextLimit) * 100).toFixed(1)}%. Applying fallback truncation.`
      );
      condensed = this.applySmartTruncation(condensed, postCondensePrediction, modelId);
    }
    return condensed;
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
    try {
      this.outputChannel.appendLine(`Sending chat request to model: ${model.id}`);
      this.outputChannel.appendLine(`Tool mode: ${options.toolMode}, Tools: ${options.tools?.length || 0}`);
      this.outputChannel.appendLine(`Message count: ${messages.length}`);

      // Resolve model variant: strip "__rb:<N>" suffix, extract per-variant budget
      const { realId: actualModelId, variantBudget } = parseModelVariant(model.id);
      if (variantBudget !== null) {
        this.outputChannel.appendLine(`Model variant resolved: realId=${actualModelId}, variantBudget=${variantBudget}`);
      }

      this.showWelcomeNotification(model.id);

      // Convert messages
      let openAIMessages: Record<string, unknown>[] = [];
      for (const msg of messages) {
        openAIMessages.push(...this.convertSingleMessageWithLogging(msg));
      }
      openAIMessages = this.sanitizePromptMessages(openAIMessages);

      const combinedSystemPrompt = await this.buildCombinedSystemPrompt(openAIMessages);
      if (combinedSystemPrompt) {
        openAIMessages.unshift({ role: 'system', content: combinedSystemPrompt });
        this.outputChannel.appendLine(`Prepended combined system prompt (${combinedSystemPrompt.length} chars)`);
      }

      // Adapt messages to the detected llama.cpp chat template before sending,
      // so 'tool' / 'system' roles are not silently mangled by an incompatible template.
      if (this.detectedBackend === 'llamacpp') {
        openAIMessages = this.adaptMessagesForTemplate(openAIMessages);
      }

      openAIMessages = this.ensureConversationHasUserTurn(openAIMessages, openAIMessages);
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

      // Calculate context usage percentage for condensation check
      const usagePercent = (overflowPrediction.estimatedTotalTokens / overflowPrediction.contextLimit) * 100;
      const condensationThreshold = this.config.contextCondensationThreshold || 80;
      const shouldAttemptCondensation = openAIMessages.length > 8
        && (usagePercent >= condensationThreshold || overflowPrediction.willOverflow);

      if (shouldAttemptCondensation) {
        const condensationReason = usagePercent >= condensationThreshold
          ? `Context at ${usagePercent.toFixed(1)}% (threshold: ${condensationThreshold}%). Attempting LLM condensation...`
          : `Context overflow predicted at ${usagePercent.toFixed(1)}% before the configured condensation threshold ${condensationThreshold}%. Attempting LLM condensation anyway...`;
        this.outputChannel.appendLine(condensationReason);
        messagesToSend = await this.condenseAndTrimIfNeeded(openAIMessages, model.id, options.tools || [], token, 'Post-condensation');
      } else if (overflowPrediction.warningLevel !== 'none' && this.config.enableProactiveTruncation !== false) {
        this.outputChannel.appendLine(`Applying proactive compaction: ${overflowPrediction.recommendedAction}`);
        messagesToSend = this.applySmartTruncation(openAIMessages, overflowPrediction, model.id);
      }

      messagesToSend = this.ensureConversationHasUserTurn(messagesToSend, openAIMessages, overflowPrediction.contextLimit);

      const truncatedMessages = messagesToSend;
      if (truncatedMessages.length < openAIMessages.length) {
        this.outputChannel.appendLine(`WARNING: Truncated conversation from ${openAIMessages.length} to ${truncatedMessages.length} messages to fit context limit`);
      }

      // Build input text for token estimation
      const inputText = this.buildInputText(truncatedMessages);

      const toolsOverhead = this.estimateToolsTokens(options.tools as readonly any[] | undefined);
      const estimatedInputTokens = await this.provideTokenCount(model, inputText, token);
      const safeMaxOutputTokens = this.calculateSafeMaxOutputTokens(model.id, estimatedInputTokens, toolsOverhead);
      const modelMaxContext = overflowPrediction.contextLimit;

      this.outputChannel.appendLine(
        `Token estimate: input=${estimatedInputTokens}, tools=${toolsOverhead}, model_context=${modelMaxContext}, chosen_max_tokens=${safeMaxOutputTokens}`
      );

      // Parse inline overrides from the last user message: /temp 0.2, /preset codegen, /grammar {...}
      const inlineOverrides = this.parseInlineOverrides(truncatedMessages);
      if (inlineOverrides.preset) {
        const overridePreset = this.loadedPresets[inlineOverrides.preset];
        if (overridePreset) {
          this.outputChannel.appendLine(`Inline override: preset=${inlineOverrides.preset}`);
        } else {
          this.outputChannel.appendLine(`WARNING: Inline /preset '${inlineOverrides.preset}' not found, ignoring`);
          inlineOverrides.preset = undefined;
        }
      }

      // Build request with dynamic temperature from preset
      const hasTools = this.config.enableToolCalling && options.tools && options.tools.length > 0;
      // Inline /preset override takes priority over config activePreset
      const effectivePresetKey = inlineOverrides.preset ?? this.config.activePreset;
      const activePreset = this.loadedPresets[effectivePresetKey] ?? this.getActivePreset();
      const presetParams = activePreset
        ? filterParamsForBackend(activePreset.params, this.detectedBackend)
        : {};

      // Determine temperature: tool mode uses agentTemperature, otherwise inline → preset → modelOptions → 0.7
      let temperature = 0.7;
      if (hasTools) {
        temperature = this.config.agentTemperature ?? 0;
      } else if (inlineOverrides.temperature !== undefined) {
        temperature = inlineOverrides.temperature;
      } else if (activePreset && activePreset.params.temperature !== undefined) {
        temperature = activePreset.params.temperature;
      } else if (options.modelOptions && typeof options.modelOptions.temperature === 'number') {
        temperature = options.modelOptions.temperature;
      }

      const requestOptions: Record<string, unknown> = {
        model: actualModelId,  // strip variant suffix before sending to server
        messages: truncatedMessages,
        max_tokens: safeMaxOutputTokens,
        temperature,
      };

      // Apply preset sampling params (backend-filtered)
      for (const [key, value] of Object.entries(presetParams)) {
        if (key !== 'temperature' && value !== undefined) {
          requestOptions[key] = value;
        }
      }
      if (activePreset) {
        this.outputChannel.appendLine(`Applied preset '${activePreset.name}' (backend: ${this.detectedBackend}): ${JSON.stringify(presetParams)}`);
      }

      // Resolve reasoning_budget: model variant > global config setting > preset (already applied above)
      // Priority: variantBudget (from model picker) > config.reasoningBudget > preset's reasoning_budget
      const effectiveBudget = variantBudget !== null ? variantBudget : this.config.reasoningBudget;
      if (effectiveBudget !== null && effectiveBudget !== undefined) {
        requestOptions['reasoning_budget'] = effectiveBudget;
        this.outputChannel.appendLine(`Reasoning budget: ${effectiveBudget} (source: ${variantBudget !== null ? 'model variant' : 'global config'})`);
      }

      // llama.cpp reasoning/thinking support:
      // - reasoning_format: "deepseek" tells the server to stream <think> content as separate
      //   `reasoning_content` field in SSE delta chunks (instead of mixing into content).
      // - chat_template_kwargs: {"enable_thinking": true} asks the Jinja template to activate
      //   thinking mode (for models like Qwen3 that support togglable thinking via template kwarg).
      // reasoningFormat setting: 'auto' = use template detection, 'deepseek' = force on, 'none' = force off
      // In 'auto' mode: if a reasoning budget is explicitly set (variant or config), we also enable
      // reasoning_format because the user clearly wants thinking — even if the template wasn't auto-detected.
      if (this.detectedBackend === 'llamacpp') {
        const formatSetting = this.config.reasoningFormat || 'auto';
        const templateSupports = this.detectedTemplate.hasNativeThinking;
        const hasBudget = effectiveBudget !== null && effectiveBudget !== undefined;
        const budgetEnablesThinking = hasBudget && effectiveBudget !== 0;
        const budgetDisablesThinking = hasBudget && effectiveBudget === 0;

        // Determine whether reasoning_format should be enabled:
        // 1. Setting = 'deepseek' → always enable
        // 2. Setting = 'auto' + template supports → enable
        // 3. Setting = 'auto' + reasoning budget explicitly set (>0 or -1) → enable (user clearly wants thinking)
        const shouldEnable = formatSetting === 'deepseek'
          || (formatSetting === 'auto' && templateSupports)
          || (formatSetting === 'auto' && budgetEnablesThinking);

        if (shouldEnable && !budgetDisablesThinking) {
          requestOptions['reasoning_format'] = 'deepseek';
          requestOptions['chat_template_kwargs'] = { enable_thinking: true };
          const source = formatSetting === 'deepseek' ? 'forced by setting'
            : templateSupports ? 'auto-detected from template'
              : 'inferred from reasoning budget';
          this.outputChannel.appendLine(`Reasoning format: deepseek (${source})`);
        } else if (budgetDisablesThinking) {
          // Budget = 0 means user explicitly disabled thinking
          requestOptions['reasoning_format'] = 'none';
          requestOptions['chat_template_kwargs'] = { enable_thinking: false };
          this.outputChannel.appendLine('Reasoning format: none (thinking disabled by budget=0)');
        } else if (formatSetting === 'none') {
          requestOptions['reasoning_format'] = 'none';
          this.outputChannel.appendLine('Reasoning format: none (forced by setting)');
        }
        // formatSetting='auto' && !templateSupports && no explicit budget: don't send anything
      }

      const toolsConfig = this.buildToolsConfig(options);
      if (toolsConfig) {
        requestOptions.tools = toolsConfig;
        if (options.toolMode !== undefined) {
          requestOptions.tool_choice = options.toolMode === vscode.LanguageModelChatToolMode.Required ? 'required' : 'auto';
        }
        requestOptions.parallel_tool_calls = this.config.parallelToolCalling;
        this.outputChannel.appendLine(`Sending ${toolsConfig.length} tools to model (parallel: ${this.config.parallelToolCalling})`);
      }

      // Allow modelOptions to override preset params (backend-aware)
      if (options.modelOptions) {
        const allowedKeys = getAllowedKeysForBackend(this.detectedBackend);
        for (const key of allowedKeys) {
          if (key in options.modelOptions && (options.modelOptions as Record<string, unknown>)[key] !== undefined) {
            (requestOptions as Record<string, unknown>)[key] = (options.modelOptions as Record<string, unknown>)[key];
          }
        }
      }

      // Apply inline /grammar or /schema override (llama.cpp only)
      if (inlineOverrides.grammar && this.detectedBackend === 'llamacpp') {
        requestOptions.grammar = inlineOverrides.grammar;
        this.outputChannel.appendLine(`Inline override: grammar applied (${inlineOverrides.grammar.length} chars)`);
      }
      if (inlineOverrides.jsonSchema) {
        requestOptions.response_format = { type: 'json_schema', json_schema: { name: 'response', schema: inlineOverrides.jsonSchema } };
        this.outputChannel.appendLine(`Inline override: JSON schema applied`);
      }

      // Store conversation snapshot for export
      this.lastConversation = {
        modelId: model.id,
        messages: truncatedMessages
          .filter((m) => typeof m.role === 'string' && typeof m.content === 'string')
          .map((m) => ({ role: m.role as string, content: m.content as string })),
      };

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
    } catch (error) {
      this.handleChatError(error, progress, token);
    }
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
      let recoveredTaggedToolCalls = 0;
      let reasoningContent = '';
      let reasoningHandled = false;
      const streamStartMs = Date.now();
      let totalOutputChars = 0;     // all generated chars (content + reasoning)
      let totalReasoningChars = 0;  // reasoning-only chars
      const idleTimeout = this.config.streamingIdleTimeout;
      const firstChunkTimeout = this.config.requestTimeout;
      let hasReceivedFirstChunk = false;
      const iterator = this.client.streamChatCompletion(
        requestOptions as unknown as OpenAIChatCompletionRequest,
        token,
        this.config.maxRetries,
        this.config.retryDelay
      )[Symbol.asyncIterator]();

      try {
        while (true) {
          if (token.isCancellationRequested) { break; }

          let result: IteratorResult<any>;
          const effectiveTimeout = hasReceivedFirstChunk ? idleTimeout : firstChunkTimeout;
          if (effectiveTimeout > 0) {
            let timeoutId: ReturnType<typeof setTimeout> | undefined;
            const timeoutErrorName = hasReceivedFirstChunk ? 'Streaming idle timeout' : 'Streaming first chunk timeout';
            const timeoutPromise = new Promise<never>((_, reject) => {
              timeoutId = setTimeout(() => reject(new Error(timeoutErrorName)), effectiveTimeout);
            });
            try {
              result = await Promise.race([iterator.next(), timeoutPromise]);
            } catch (e: any) {
              if (e.message === 'Streaming idle timeout') {
                this.outputChannel.appendLine(`WARNING: Streaming idle timeout after ${effectiveTimeout}ms`);
                break;
              }
              if (e.message === 'Streaming first chunk timeout') {
                this.outputChannel.appendLine(`WARNING: Streaming first chunk timeout after ${effectiveTimeout}ms`);
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
          hasReceivedFirstChunk = true;
          const chunk = result.value;

          if (chunk.reasoning_content) {
            reasoningContent += chunk.reasoning_content;
            totalReasoningChars += chunk.reasoning_content.length;
            totalOutputChars += chunk.reasoning_content.length;
          }

          const reasoningContainsTaggedToolCall = reasoningContent.includes('<tool_call>');
          if (!reasoningHandled && reasoningContent && !reasoningContainsTaggedToolCall && (chunk.content || chunk.finished_tool_calls?.length)) {
            const normalizedReasoning = this.normalizeReasoningContent(reasoningContent);
            reasoningHandled = true;
            if (normalizedReasoning) {
              if (this.config.showThinking) {
                // Show thinking as blockquote (VS Code chat doesn't render HTML)
                const thinkingLines = normalizedReasoning.split('\n').map(line => `> ${line}`).join('\n');
                const thinkingSection = `> **💭 Thinking** *(${normalizedReasoning.length} chars)*\n${thinkingLines}\n\n---\n\n`;
                progress.report(new vscode.LanguageModelTextPart(thinkingSection));
                this.outputChannel.appendLine(`Displayed thinking content in chat UI (${normalizedReasoning.length} chars)`);
              } else {
                this.outputChannel.appendLine(`Suppressed reasoning content from chat UI (${normalizedReasoning.length} chars)`);
              }
            }
          }

          if (chunk.content) {
            const visibleContent = this.sanitizeVisibleContent(chunk.content);
            if (visibleContent) {
              totalContent += visibleContent;
              totalOutputChars += visibleContent.length;
              progress.report(new vscode.LanguageModelTextPart(visibleContent));
            }
          }

          if (chunk.finished_tool_calls?.length) {
            for (const toolCall of chunk.finished_tool_calls) {
              totalToolCalls++;
              this.processToolCall(toolCall, progress);
            }
          }
        }
      } finally {
        totalToolCalls += await this.closeAndDrainStreamIterator(iterator, progress, !token.isCancellationRequested);
      }

      if (totalToolCalls === 0) {
        const extractedReasoningCalls = this.getRecoverableTaggedToolCalls(reasoningContent, 'reasoning_content');
        if (extractedReasoningCalls) {
          reasoningContent = extractedReasoningCalls.remainingText;
          this.outputChannel.appendLine(`Recovered ${extractedReasoningCalls.toolCalls.length} tagged tool call(s) from reasoning_content fallback`);
          for (const toolCall of extractedReasoningCalls.toolCalls) {
            totalToolCalls++;
            recoveredTaggedToolCalls++;
            this.processToolCall(toolCall, progress);
          }
        }

        if (totalToolCalls === 0) {
          const extractedContentCalls = this.getRecoverableTaggedToolCalls(totalContent, 'content');
          if (extractedContentCalls) {
            totalContent = extractedContentCalls.remainingText;
            this.outputChannel.appendLine(`Recovered ${extractedContentCalls.toolCalls.length} tagged tool call(s) from content fallback`);
            for (const toolCall of extractedContentCalls.toolCalls) {
              totalToolCalls++;
              recoveredTaggedToolCalls++;
              this.processToolCall(toolCall, progress);
            }
          }
        }
      }

      const isReasoningOnlyEmptyResponse =
        totalContent.length === 0
        && totalToolCalls === 0
        && reasoningContent.trim().length > 0;

      const hasToolsInRequest = Array.isArray(requestOptions.tools) && requestOptions.tools.length > 0;
      const reasoningFormat = typeof requestOptions.reasoning_format === 'string'
        ? requestOptions.reasoning_format
        : undefined;

      if (
        isReasoningOnlyEmptyResponse
        && retryAttempt < 1
        && this.detectedBackend === 'llamacpp'
        && hasToolsInRequest
        && reasoningFormat === 'deepseek'
      ) {
        this.outputChannel.appendLine(
          'Reasoning-only response with no content/tool calls detected on llama.cpp. Retrying once without a separate reasoning stream so thinking stays enabled but the model must produce visible content or tool calls.'
        );

        const retryOptions: Record<string, unknown> = {
          ...requestOptions,
          chat_template_kwargs: {
            ...(
              typeof requestOptions.chat_template_kwargs === 'object' && requestOptions.chat_template_kwargs !== null
                ? requestOptions.chat_template_kwargs as Record<string, unknown>
                : {}
            ),
            enable_thinking: true,
          },
        };

        delete retryOptions.reasoning_format;

        return this.executeStreamWithRetry(
          retryOptions,
          model,
          openAIMessages,
          options,
          token,
          progress,
          inputText,
          retryAttempt + 1
        );
      }

      if (!reasoningHandled && reasoningContent) {
        const normalizedReasoning = this.normalizeReasoningContent(reasoningContent);
        reasoningHandled = true;
        if (normalizedReasoning) {
          if (this.config.showThinking) {
            const thinkingLines = normalizedReasoning.split('\n').map(line => `> ${line}`).join('\n');
            const thinkingSection = `> **💭 Thinking** *(${normalizedReasoning.length} chars)*\n${thinkingLines}\n\n---\n\n`;
            progress.report(new vscode.LanguageModelTextPart(thinkingSection));
            this.outputChannel.appendLine(`Displayed final thinking content in chat UI (${normalizedReasoning.length} chars)`);
          } else {
            this.outputChannel.appendLine(`Suppressed final reasoning content from chat UI (${normalizedReasoning.length} chars)`);
          }
        }
      }

      // Update request stats for status bar
      const elapsedSec = Math.max(0.1, (Date.now() - streamStartMs) / 1000);
      const modelCtx = model.maxInputTokens ?? this.config.defaultMaxTokens;
      // Use the actual estimated input tokens we computed earlier (estimatedInputTokens from provideTokenCount)
      const inputTokens = Math.ceil((this.buildInputText(Array.isArray(requestOptions.messages) ? requestOptions.messages as Record<string, unknown>[] : [])).length / 3.3);
      // Token count: total generated chars / 3.3 (matches our input estimation ratio)
      const outputTokenCount = Math.ceil(totalOutputChars / 3.3);
      const tokPerSec = Math.round(outputTokenCount / elapsedSec);
      const ctxPercent = modelCtx > 0 ? Math.round((inputTokens / modelCtx) * 100) : 0;
      const reasoningTokens = Math.ceil(totalReasoningChars / 3.3);
      this.updateRequestStats({
        lastTokensPerSec: tokPerSec,
        lastContextPercent: ctxPercent,
        lastInputTokens: inputTokens,
        lastOutputChars: totalContent.length,
        requestCount: this.requestStats.requestCount + 1,
      });

      // Show inline stats footer in chat (only for text responses, not pure tool calls)
      if (totalContent.length > 0) {
        const presetLabel = this.getActivePreset()?.name ?? this.config.activePreset;
        const thinkingNote = reasoningTokens > 0 ? ` · ${reasoningTokens} think` : '';
        const statsLine = `\n\n---\n*${tokPerSec} t/s · ${ctxPercent}% ctx${thinkingNote} · ${presetLabel}*`;
        progress.report(new vscode.LanguageModelTextPart(statsLine));
      }

      // Append response to conversation snapshot
      if (this.lastConversation && totalContent) {
        this.lastConversation.messages.push({ role: 'assistant', content: totalContent, thinking: reasoningContent || undefined });
      }
      if (recoveredTaggedToolCalls > 0) {
        this.outputChannel.appendLine(`Tool-call fallback activated: recovered ${recoveredTaggedToolCalls} tagged call(s) from unstructured output`);
      }
      this.outputChannel.appendLine(`Completed chat request, received ${totalContent.length} chars content + ${totalReasoningChars} chars reasoning, ${totalToolCalls} tool calls, ~${outputTokenCount} output tokens (${reasoningTokens} thinking) in ${elapsedSec.toFixed(1)}s (~${tokPerSec} tok/s)`);

      if (totalContent.length === 0 && totalToolCalls === 0) {
        await this.handleEmptyResponse(model, inputText, openAIMessages.length, requestOptions.tools ? (requestOptions.tools as unknown[]).length : 0, token, progress);
      }
    } catch (error: any) {
      const missingUserQuery = typeof error?.message === 'string' && error.message.includes('No user query found in messages');
      if (missingUserQuery && retryAttempt < 1) {
        const currentMessages = Array.isArray(requestOptions.messages) ? requestOptions.messages as Record<string, unknown>[] : [];
        const contextBudget = this.modelMetadata.get(model.id)?.maxTokens || this.config.defaultMaxTokens;
        const repairedMessages = this.ensureConversationHasUserTurn(currentMessages, openAIMessages, contextBudget);

        if (JSON.stringify(repairedMessages) !== JSON.stringify(currentMessages)) {
          this.outputChannel.appendLine('Server rejected request without a user turn. Retrying once with repaired conversation history.');

          const retryTools = Array.isArray(requestOptions.tools) ? requestOptions.tools as Record<string, unknown>[] : undefined;
          const repairedInputText = this.buildInputText(repairedMessages);
          const repairedEstimate = await this.provideTokenCount(model, repairedInputText, token);
          const repairedMaxOutput = this.calculateSafeMaxOutputTokens(model.id, repairedEstimate, this.estimateToolsTokens(retryTools));

          return this.executeStreamWithRetry(
            {
              ...requestOptions,
              messages: repairedMessages,
              max_tokens: repairedMaxOutput,
            },
            model,
            openAIMessages,
            options,
            token,
            progress,
            repairedInputText,
            retryAttempt + 1
          );
        }
      }

      if (error.isContextOverflow && retryAttempt < maxContextRetries) {
        const serverPromptTokens = typeof error.promptTokens === 'number' ? error.promptTokens : undefined;
        const serverContextSize = typeof error.contextSize === 'number'
          ? error.contextSize
          : (this.modelMetadata.get(model.id)?.maxTokens || this.config.defaultMaxTokens);
        this.outputChannel.appendLine(`Context overflow from server: ${serverPromptTokens ?? 'unknown'} prompt tokens > ${serverContextSize} context. Retry ${retryAttempt + 1}/${maxContextRetries}`);

        const { realId: actualModelId } = parseModelVariant(model.id);

        if (serverContextSize && serverContextSize > 0) {
          const updatedLimits = {
            maxTokens: serverContextSize,
            maxOutputTokens: Math.min(this.config.defaultMaxOutputTokens, Math.floor(serverContextSize / 2)),
          };
          this.modelMetadata.set(model.id, updatedLimits);
          this.modelMetadata.set(actualModelId, updatedLimits);
        }

        let currentMessages = Array.isArray(requestOptions.messages) ? requestOptions.messages as any[] : [];
        if (currentMessages.length > 8) {
          this.outputChannel.appendLine('Attempting LLM condensation during context-overflow recovery before hard truncation.');
          currentMessages = await this.condenseAndTrimIfNeeded(currentMessages, model.id, options.tools || [], token, 'Retry post-condensation');
        }
        const reserveTokens = Math.max(256, Math.ceil(serverContextSize * 0.05));
        let retryTools = Array.isArray(requestOptions.tools) ? requestOptions.tools as Record<string, unknown>[] : undefined;
        let toolTokens = this.estimateToolsTokens(retryTools);
        const targetPromptTokens = Math.max(512, serverContextSize - reserveTokens);
        let messageBudget = Math.max(256, targetPromptTokens - toolTokens);
        let aggressivelyTruncated = this.ensureConversationHasUserTurn(
          this.truncateMessagesToFit(currentMessages, messageBudget),
          openAIMessages,
          messageBudget
        );
        let retryPromptEstimate = this.estimatePromptTokens(aggressivelyTruncated, retryTools);

        this.outputChannel.appendLine(
          `Retry budgeting: prompt_target=${targetPromptTokens}, message_budget=${messageBudget}, tool_tokens=${toolTokens}, estimate=${retryPromptEstimate}`
        );

        if (retryPromptEstimate > targetPromptTokens && retryTools && options.toolMode !== vscode.LanguageModelChatToolMode.Required) {
          this.outputChannel.appendLine(`Retry still too large. Disabling tools to recover ~${toolTokens} prompt tokens.`);
          retryTools = undefined;
          toolTokens = 0;
          messageBudget = targetPromptTokens;
          aggressivelyTruncated = this.ensureConversationHasUserTurn(
            this.truncateMessagesToFit(currentMessages, messageBudget),
            openAIMessages,
            messageBudget
          );
          retryPromptEstimate = this.estimatePromptTokens(aggressivelyTruncated, retryTools);
        }

        if (retryPromptEstimate > targetPromptTokens) {
          const tighterBudget = Math.max(128, messageBudget - Math.max(256, retryPromptEstimate - targetPromptTokens));
          this.outputChannel.appendLine(`Applying tighter retry truncation: ${messageBudget} -> ${tighterBudget} message tokens`);
          aggressivelyTruncated = this.ensureConversationHasUserTurn(
            this.truncateMessagesToFit(aggressivelyTruncated, tighterBudget),
            openAIMessages,
            tighterBudget
          );
          retryPromptEstimate = this.estimatePromptTokens(aggressivelyTruncated, retryTools);
        }

        this.outputChannel.appendLine(
          `Aggressive truncation: ${currentMessages.length} -> ${aggressivelyTruncated.length} messages, estimated prompt ${retryPromptEstimate}/${targetPromptTokens}`
        );

        const newInputText = this.buildInputText(aggressivelyTruncated);
        const newEstimate = await this.provideTokenCount(model, newInputText, token);
        const newMaxOutput = this.calculateSafeMaxOutputTokens(model.id, newEstimate, toolTokens);

        const retryOptions: Record<string, unknown> = {
          ...requestOptions,
          messages: aggressivelyTruncated,
          max_tokens: newMaxOutput,
        };

        if (retryTools) {
          retryOptions.tools = retryTools;
        } else {
          delete retryOptions.tools;
          delete retryOptions.tool_choice;
          delete retryOptions.parallel_tool_calls;
        }

        return this.executeStreamWithRetry(retryOptions, model, openAIMessages, options, token, progress, newInputText, retryAttempt + 1);
      }

      throw error;
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

    const requestTimeout = config.get<number>('requestTimeout', 300000);
    const streamingIdleTimeoutInspect = config.inspect<number>('streamingIdleTimeout');
    const hasExplicitStreamingIdleTimeout =
      streamingIdleTimeoutInspect?.globalValue !== undefined ||
      streamingIdleTimeoutInspect?.workspaceValue !== undefined ||
      streamingIdleTimeoutInspect?.workspaceFolderValue !== undefined;
    let streamingIdleTimeout = config.get<number>('streamingIdleTimeout', 120000);

    // Keep streaming idle timeout aligned with request timeout unless the user explicitly overrides it.
    if (!hasExplicitStreamingIdleTimeout) {
      streamingIdleTimeout = requestTimeout;
    }

    const cfg: GatewayConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      apiKey: config.get<string>('apiKey', ''),
      requestTimeout,
      defaultMaxTokens: config.get<number>('defaultMaxTokens', 32768),
      defaultMaxOutputTokens: config.get<number>('defaultMaxOutputTokens', 4096),
      enableToolCalling: config.get<boolean>('enableToolCalling', true),
      parallelToolCalling: config.get<boolean>('parallelToolCalling', true),
      agentTemperature: config.get<number>('agentTemperature', 0),
      toolExcludePatterns: config.get<string[]>('toolExcludePatterns', ['^search_view_results$', '^view_results$', '^vscode_internal', '^extension_', '^unknown_']),
      streamingIdleTimeout,
      maxRetries: config.get<number>('maxRetries', 2),
      retryDelay: config.get<number>('retryDelay', 1000),
      systemPrompt: config.get<string>('systemPrompt', ''),
      promptStripPatterns: config.get<string[]>('promptStripPatterns', DEFAULT_PROMPT_STRIP_PATTERNS),
      contextWarningThreshold: config.get<number>('contextWarningThreshold', 75),
      contextHardLimit: config.get<number>('contextHardLimit', 85),
      maxMessageHistory: config.get<number>('maxMessageHistory', 50),
      enableProactiveTruncation: config.get<boolean>('enableProactiveTruncation', true),
      activePreset: config.get<string>('activePreset', 'codegen'),
      showThinking: config.get<boolean>('showThinking', false),
      reasoningBudget: config.get<number | null>('reasoningBudget', null),
      reasoningFormat: config.get<'auto' | 'deepseek' | 'none'>('reasoningFormat', 'auto'),
      contextCondensationThreshold: config.get<number>('contextCondensationThreshold', 80),
    };

    // Validate requestTimeout
    if (cfg.requestTimeout <= 0) {
      this.outputChannel.appendLine(`ERROR: requestTimeout must be > 0; using default 60000`);
      cfg.requestTimeout = 60000;
    }

    // If unset, keep stream idle timeout in sync with the validated request timeout.
    if (!hasExplicitStreamingIdleTimeout) {
      cfg.streamingIdleTimeout = cfg.requestTimeout;
      this.outputChannel.appendLine(
        `INFO: github.copilot.llm-gateway.streamingIdleTimeout is not explicitly set; using requestTimeout (${cfg.requestTimeout}ms).`
      );
    }

    if (cfg.streamingIdleTimeout < 0) {
      this.outputChannel.appendLine(`ERROR: streamingIdleTimeout must be >= 0; using requestTimeout (${cfg.requestTimeout}ms)`);
      cfg.streamingIdleTimeout = cfg.requestTimeout;
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

  /**
   * Get the currently active sampling preset from loaded file-based presets.
   */
  getActivePreset(): SamplingPreset | undefined {
    const name = this.config.activePreset || 'codegen';
    return this.loadedPresets[name];
  }

  /**
   * Initialize file-based preset system: ensure dir, load presets, set up file watcher.
   */
  private initializePresets(): void {
    const configuredPath = vscode.workspace.getConfiguration('github.copilot.llm-gateway')
      .get<string>('presetsPath', '');
    this.presetsDir = resolvePresetsDir(configuredPath);

    try {
      ensurePresetsDir(this.presetsDir);
      this.reloadPresetsFromDisk();
      this.outputChannel.appendLine(`Presets directory: ${this.presetsDir} (${Object.keys(this.loadedPresets).length} presets loaded)`);
    } catch (err) {
      this.outputChannel.appendLine(`ERROR: Failed to initialize presets directory: ${err}`);
    }

    // Watch for preset file changes (create/modify/delete)
    this.setupPresetWatcher();
  }

  /**
   * Reload all presets from disk.
   */
  private reloadPresetsFromDisk(): void {
    this.loadedPresets = loadPresetsFromDir(
      this.presetsDir,
      (msg: string) => this.outputChannel.appendLine(msg)
    );
    this.updatePresetStatusBar();
  }

  /**
   * Set up a file system watcher on the presets directory for live reload.
   */
  private setupPresetWatcher(): void {
    this.presetWatcher?.dispose();

    const pattern = new vscode.RelativePattern(vscode.Uri.file(this.presetsDir), '*.json');
    this.presetWatcher = vscode.workspace.createFileSystemWatcher(pattern);

    const reload = () => {
      this.outputChannel.appendLine('Preset file changed, reloading presets...');
      this.reloadPresetsFromDisk();
    };
    this.presetWatcher.onDidChange(reload);
    this.presetWatcher.onDidCreate(reload);
    this.presetWatcher.onDidDelete(reload);

    this.context.subscriptions.push(this.presetWatcher);
  }

  /**
   * Show quick pick for preset selection.
   */
  async showPresetPicker(): Promise<void> {
    const allPresets = this.loadedPresets;
    const currentPreset = this.config.activePreset || 'codegen';

    interface PresetQuickPickItem extends vscode.QuickPickItem {
      preset?: string;
      action?: string;
    }

    const items: PresetQuickPickItem[] = Object.entries(allPresets).map(([key, preset]) => ({
      label: `${key === currentPreset ? '$(check) ' : ''}${preset.name}`,
      description: key === currentPreset ? '(active)' : '',
      detail: `${preset.description}`,
      preset: key,
    }));

    // Add separator and management actions
    items.push({ label: '', kind: vscode.QuickPickItemKind.Separator });
    items.push({
      label: '$(folder-opened) Open Presets Folder',
      detail: this.presetsDir,
      action: 'openFolder',
    });
    items.push({
      label: '$(add) Create New Preset',
      detail: 'Create a new preset JSON file from a template',
      action: 'create',
    });

    const pick = await vscode.window.showQuickPick(items, {
      title: `Select Sampling Preset (backend: ${this.detectedBackend})`,
      placeHolder: 'Choose a preset, or open the folder to edit',
    });

    if (!pick) return;

    if (pick.action === 'openFolder') {
      await this.openPresetsFolder();
    } else if (pick.action === 'create') {
      await this.createNewPreset();
    } else if (pick.preset) {
      await vscode.workspace.getConfiguration('github.copilot.llm-gateway')
        .update('activePreset', pick.preset, vscode.ConfigurationTarget.Global);
      this.outputChannel.appendLine(`Preset changed to: ${pick.preset}`);
    }
  }

  /**
   * Show a Quick Pick for selecting the global reasoning/thinking token budget.
   * null = use preset/server default, 0 = off, -1 = unlimited, >0 = token limit.
   */
  async showReasoningBudgetPicker(): Promise<void> {
    const current = this.config.reasoningBudget;

    interface BudgetItem extends vscode.QuickPickItem {
      value: number | null;
    }

    const options: BudgetItem[] = [
      {
        label: `${current === null ? '$(check) ' : ''}Server / Preset Default`,
        description: current === null ? '(active)' : '',
        detail: 'Do not send reasoning_budget — let the preset or server decide',
        value: null,
      },
      {
        label: `${current === 0 ? '$(check) ' : ''}Off (0)`,
        description: current === 0 ? '(active)' : '',
        detail: 'Disable thinking entirely (reasoning_budget=0)',
        value: 0,
      },
      {
        label: `${current === 1024 ? '$(check) ' : ''}Short (1 024 tokens)`,
        description: current === 1024 ? '(active)' : '',
        detail: 'Quick reasoning pass — suitable for simple tasks',
        value: 1024,
      },
      {
        label: `${current === 4096 ? '$(check) ' : ''}Medium (4 096 tokens)`,
        description: current === 4096 ? '(active)' : '',
        detail: 'Balanced reasoning depth',
        value: 4096,
      },
      {
        label: `${current === 16384 ? '$(check) ' : ''}Long (16 384 tokens)`,
        description: current === 16384 ? '(active)' : '',
        detail: 'Deep reasoning for complex problems',
        value: 16384,
      },
      {
        label: `${current === -1 ? '$(check) ' : ''}Unlimited (-1)`,
        description: current === -1 ? '(active)' : '',
        detail: 'No token cap on thinking (may be slow for large problems)',
        value: -1,
      },
    ];

    const pick = await vscode.window.showQuickPick(options, {
      title: 'Select Reasoning Budget',
      placeHolder: 'Thinking token limit (llama.cpp reasoning_budget)',
    }) as BudgetItem | undefined;

    if (!pick) return;

    await vscode.workspace.getConfiguration('github.copilot.llm-gateway')
      .update('reasoningBudget', pick.value, vscode.ConfigurationTarget.Global);

    const label = pick.value === null
      ? 'Server / Preset Default'
      : pick.value === 0 ? 'Off'
        : pick.value === -1 ? 'Unlimited'
          : `${pick.value} tokens`;
    vscode.window.showInformationMessage(`LLM Gateway: Reasoning budget set to ${label}`);
    this.outputChannel.appendLine(`Reasoning budget changed to: ${pick.value}`);
  }

  /**
   * Open the presets folder in VS Code file explorer and reveal the active preset file.
   */
  async openPresetsFolder(): Promise<void> {
    const activeKey = this.config.activePreset || 'codegen';
    const filePath = getPresetFilePath(this.presetsDir, activeKey);
    try {
      const doc = await vscode.workspace.openTextDocument(filePath);
      await vscode.window.showTextDocument(doc);
    } catch {
      // If the active preset file doesn't exist, just open the folder
      const uri = vscode.Uri.file(this.presetsDir);
      await vscode.commands.executeCommand('revealFileInOS', uri);
    }
  }

  /**
   * Create a new preset from a template.
   */
  async createNewPreset(): Promise<void> {
    const name = await vscode.window.showInputBox({
      prompt: 'Preset key (lowercase, no spaces — becomes the filename)',
      placeHolder: 'e.g. my-custom-preset',
      validateInput: (val: string) => {
        if (!val.trim()) return 'Name is required';
        if (!/^[a-z0-9_-]+$/.test(val)) return 'Only lowercase letters, numbers, hyphens, underscores';
        if (this.loadedPresets[val]) return `Preset '${val}' already exists`;
        return null;
      },
    });
    if (!name) return;

    const displayName = await vscode.window.showInputBox({
      prompt: 'Display name for the preset',
      placeHolder: 'e.g. My Custom Preset',
      value: name.charAt(0).toUpperCase() + name.slice(1),
    });
    if (!displayName) return;

    const template: SamplingPreset = {
      name: displayName,
      description: 'Custom preset — edit params below',
      backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
      params: {
        temperature: 0.3,
        top_p: 0.9,
        top_k: 30,
        min_p: 0.04,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
      },
    };

    const filePath = getPresetFilePath(this.presetsDir, name);
    const content = JSON.stringify(template, null, 2) + '\n';
    const { writeFileSync } = require('node:fs');
    writeFileSync(filePath, content, 'utf-8');

    // Open the file for editing
    const doc = await vscode.workspace.openTextDocument(filePath);
    await vscode.window.showTextDocument(doc);

    vscode.window.showInformationMessage(`Preset '${name}' created. Edit the file and save — it will auto-reload.`);
  }

  /**
   * Update status bar to show current preset and backend.
   */
  private updatePresetStatusBar(): void {
    if (!this.presetStatusBarItem) {
      this.presetStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 99);
      this.presetStatusBarItem.command = 'github.copilot.llm-gateway.selectPreset';
      this.context.subscriptions.push(this.presetStatusBarItem);
    }
    const preset = this.getActivePreset();
    const presetName = preset?.name ?? this.config.activePreset;
    this.presetStatusBarItem.text = `$(settings-gear) ${presetName}`;
    this.presetStatusBarItem.tooltip = `LLM Gateway Preset: ${presetName} (${this.detectedBackend}) — Click to change`;
    this.presetStatusBarItem.show();
  }

  private dispose(): void {
    this.statusBarItem?.dispose();
    this.presetStatusBarItem?.dispose();
    this.presetWatcher?.dispose();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Inline override parser: /temp 0.2  /preset codegen  /grammar {...}
  // ──────────────────────────────────────────────────────────────────────────
  private parseInlineOverrides(messages: Record<string, unknown>[]): {
    temperature?: number;
    preset?: string;
    grammar?: string;
    jsonSchema?: Record<string, unknown>;
  } {
    const result: { temperature?: number; preset?: string; grammar?: string; jsonSchema?: Record<string, unknown> } = {};
    // Only look at the last user message
    const last = [...messages].reverse().find((m) => m.role === 'user' && typeof m.content === 'string');
    if (!last || typeof last.content !== 'string') return result;

    const text = last.content;

    // /temp 0.2 or /temperature 0.2
    const tempMatch = text.match(/\/(?:temp|temperature)\s+([0-9]*\.?[0-9]+)/i);
    if (tempMatch) {
      const val = parseFloat(tempMatch[1]);
      if (!isNaN(val) && val >= 0 && val <= 2) result.temperature = val;
    }

    // /preset codegen
    const presetMatch = text.match(/\/preset\s+([a-z0-9_-]+)/i);
    if (presetMatch) result.preset = presetMatch[1].toLowerCase();

    // /grammar { ... } — rest of line is GBNF grammar string
    const grammarMatch = text.match(/\/grammar\s+(.+)$/ms);
    if (grammarMatch) result.grammar = grammarMatch[1].trim();

    // /schema { ... } — JSON schema for structured output
    const schemaMatch = text.match(/\/schema\s+(\{[\s\S]+\})/);
    if (schemaMatch) {
      try { result.jsonSchema = JSON.parse(schemaMatch[1]); } catch { /* ignore invalid */ }
    }

    return result;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Stats: update stored stats + status bar
  // ──────────────────────────────────────────────────────────────────────────
  private updateRequestStats(stats: RequestStats): void {
    this.requestStats = stats;
    this.updateStatsStatusBar();
  }

  private updateStatsStatusBar(): void {
    if (!this.statusBarItem) return;
    const { lastTokensPerSec, lastContextPercent, requestCount } = this.requestStats;
    const ctxIcon = lastContextPercent >= 80 ? '$(warning)' : '$(pulse)';
    this.statusBarItem.text = `$(hubot) ${lastTokensPerSec > 0 ? `${lastTokensPerSec}t/s` : 'LLM'} ${ctxIcon}${lastContextPercent}%`;
    this.statusBarItem.tooltip = [
      `LLM Gateway — ${requestCount} requests`,
      `Last: ~${lastTokensPerSec} tokens/sec`,
      `Context: ${lastContextPercent}% used`,
      `Input tokens: ${this.requestStats.lastInputTokens}`,
      `Output chars: ${this.requestStats.lastOutputChars}`,
      'Click to open settings',
    ].join('\n');
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Model stats command
  // ──────────────────────────────────────────────────────────────────────────
  async showModelStats(): Promise<void> {
    const stats = this.requestStats;
    const preset = this.getActivePreset();
    const info = [
      `**LLM Gateway — Model Stats**`,
      ``,
      `| | |`,
      `|---|---|`,
      `| Server | ${this.config.serverUrl} |`,
      `| Backend | ${this.detectedBackend} |`,
      `| Active Preset | ${preset?.name ?? this.config.activePreset} |`,
      `| Total Requests | ${stats.requestCount} |`,
      `| Last Speed | ~${stats.lastTokensPerSec} tokens/sec |`,
      `| Last Context Use | ${stats.lastContextPercent}% |`,
      `| Last Input Tokens | ${stats.lastInputTokens} |`,
      `| Last Output Chars | ${stats.lastOutputChars} |`,
    ].join('\n');

    // Show as info message with "Open Output" action
    const action = await vscode.window.showInformationMessage(
      `LLM Gateway: ${stats.requestCount} requests | last ~${stats.lastTokensPerSec}t/s | ctx ${stats.lastContextPercent}%`,
      'Open Output Log'
    );
    if (action === 'Open Output Log') {
      this.outputChannel.show();
    }
    this.outputChannel.appendLine(info.replace(/\*\*/g, '').replace(/\|/g, '|'));
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Export conversation
  // ──────────────────────────────────────────────────────────────────────────
  async exportConversation(): Promise<void> {
    if (!this.lastConversation || this.lastConversation.messages.length === 0) {
      vscode.window.showWarningMessage('LLM Gateway: No conversation to export yet.');
      return;
    }

    const { modelId, messages } = this.lastConversation;
    const date = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const preset = this.getActivePreset();

    const lines: string[] = [
      `# LLM Gateway Conversation Export`,
      ``,
      `**Date:** ${new Date().toLocaleString()}  `,
      `**Model:** ${modelId}  `,
      `**Preset:** ${preset?.name ?? this.config.activePreset}  `,
      `**Backend:** ${this.detectedBackend}  `,
      ``,
      `---`,
      ``,
    ];

    for (const msg of messages) {
      const roleLabel = msg.role === 'user' ? '👤 User' : msg.role === 'assistant' ? '🤖 Assistant' : '⚙️ System';
      lines.push(`## ${roleLabel}`, ``);
      if (msg.thinking) {
        lines.push(`> **💭 Thinking**`);
        msg.thinking.split('\n').forEach((l) => lines.push(`> ${l}`));
        lines.push(``, `---`, ``);
      }
      lines.push(msg.content, ``, `---`, ``);
    }

    const content = lines.join('\n');
    const defaultUri = vscode.Uri.file(`${process.env.HOME ?? '/tmp'}/llm-conversation-${date}.md`);

    const saveUri = await vscode.window.showSaveDialog({
      defaultUri,
      filters: { Markdown: ['md'], Text: ['txt'] },
      title: 'Export Conversation',
    });

    if (!saveUri) return;

    const { writeFileSync } = require('node:fs');
    writeFileSync(saveUri.fsPath, content, 'utf-8');
    const doc = await vscode.workspace.openTextDocument(saveUri);
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage(`Conversation exported to ${saveUri.fsPath}`);
  }

  private reloadConfig(): void {
    this.config = this.loadConfig();
    this.client.updateConfig(this.config);
    this.modelMetadata.clear();

    // Re-resolve presets dir if path setting changed
    const configuredPath = vscode.workspace.getConfiguration('github.copilot.llm-gateway')
      .get<string>('presetsPath', '');
    const newDir = resolvePresetsDir(configuredPath);
    if (newDir !== this.presetsDir) {
      this.presetsDir = newDir;
      ensurePresetsDir(this.presetsDir);
      this.setupPresetWatcher();
      this.outputChannel.appendLine(`Presets directory changed to: ${this.presetsDir}`);
    }
    this.reloadPresetsFromDisk();

    this.outputChannel.appendLine('Configuration reloaded, model metadata cache cleared');
  }
}
