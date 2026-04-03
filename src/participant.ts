import * as vscode from 'vscode';
import type { GatewayProvider } from './provider';

const PARTICIPANT_ID = 'llm-gateway';

export function registerChatParticipant(
    context: vscode.ExtensionContext,
    provider: GatewayProvider
): void {
    const participant = vscode.chat.createChatParticipant(PARTICIPANT_ID, (
        request: vscode.ChatRequest,
        _context: vscode.ChatContext,
        stream: vscode.ChatResponseStream,
        _token: vscode.CancellationToken
    ) => handleRequest(request, stream, provider));

    participant.iconPath = vscode.Uri.joinPath(context.extensionUri, 'assets', 'copilot-llm-gateway.png');

    context.subscriptions.push(participant);
}

async function handleRequest(
    request: vscode.ChatRequest,
    stream: vscode.ChatResponseStream,
    provider: GatewayProvider
): Promise<void> {
    const cmd = request.command;

    if (cmd === 'stats') {
        renderStats(stream, provider);
    } else if (cmd === 'presets') {
        renderPresets(stream, provider);
    } else if (cmd === 'preset') {
        await switchPreset(request.prompt.trim(), stream, provider);
    } else if (cmd === 'config') {
        renderConfig(stream, provider);
    } else if (cmd === 'changelog') {
        renderChangelog(stream);
    } else {
        renderHelp(stream);
    }
}

function renderStats(stream: vscode.ChatResponseStream, provider: GatewayProvider): void {
    const stats = provider.getStats();
    const preset = provider.getActivePreset();
    const backend = provider.getBackend();

    stream.markdown(`## LLM Gateway Stats\n\n`);
    stream.markdown(`| Metric | Value |\n|---|---|\n`);
    stream.markdown(`| Server | \`${provider.getConfig().serverUrl}\` |\n`);
    stream.markdown(`| Backend | ${backend} |\n`);
    stream.markdown(`| Active Preset | ${preset?.name ?? provider.getConfig().activePreset} |\n`);
    stream.markdown(`| Total Requests | ${stats.requestCount} |\n`);
    stream.markdown(`| Last Speed | ~${stats.lastTokensPerSec} tok/s |\n`);
    stream.markdown(`| Context Used | ${stats.lastContextPercent}% |\n`);
    stream.markdown(`| Last Input Tokens | ${stats.lastInputTokens} |\n`);
    stream.markdown(`| Last Output Chars | ${stats.lastOutputChars} |\n`);
}

function renderPresets(stream: vscode.ChatResponseStream, provider: GatewayProvider): void {
    const presets = provider.getPresets();
    const activeKey = provider.getConfig().activePreset;

    stream.markdown(`## Sampling Presets\n\n`);
    stream.markdown(`Active: **${activeKey}** | Backend: **${provider.getBackend()}**\n\n`);
    stream.markdown(`| Preset | Temp | top_p | top_k | min_p | Special |\n`);
    stream.markdown(`|--------|------|-------|-------|-------|---------|\n`);

    for (const [key, preset] of Object.entries(presets)) {
        const p = preset.params;
        const specials: string[] = [];
        if (p.mirostat) specials.push(`mirostat=${p.mirostat}`);
        if (p.dry_multiplier) specials.push(`DRY=${p.dry_multiplier}`);
        if (p.xtc_probability) specials.push(`XTC=${p.xtc_probability}`);
        if (p.dynatemp_range) specials.push(`dynatemp=${p.dynatemp_range}`);
        if (p.seed !== undefined) specials.push(`seed=${p.seed}`);
        if (p.presence_penalty) specials.push(`pres=${p.presence_penalty}`);
        if (p.frequency_penalty) specials.push(`freq=${p.frequency_penalty}`);

        const marker = key === activeKey ? '**→**' : '';
        stream.markdown(
            `| ${marker}${preset.name} | ${p.temperature ?? '-'} | ${p.top_p ?? '-'} | ${p.top_k ?? '-'} | ${p.min_p ?? '-'} | ${specials.join(', ') || '-'} |\n`
        );
    }

    stream.markdown(`\n> Switch preset: \`@llm-gateway /preset <name>\` or use **LLM Gateway: Select Sampling Preset** command.\n`);

    for (const key of Object.keys(presets)) {
        if (key !== activeKey) {
            stream.button({
                command: 'github.copilot.llm-gateway.switchPresetByName',
                arguments: [key],
                title: `Switch to ${key}`,
            });
        }
    }
}

function renderConfig(stream: vscode.ChatResponseStream, provider: GatewayProvider): void {
    const cfg = provider.getConfig();

    stream.markdown(`## LLM Gateway Configuration\n\n`);
    stream.markdown(`| Setting | Value |\n|---|---|\n`);
    stream.markdown(`| serverUrl | \`${cfg.serverUrl}\` |\n`);
    stream.markdown(`| defaultMaxTokens | ${cfg.defaultMaxTokens} |\n`);
    stream.markdown(`| defaultMaxOutputTokens | ${cfg.defaultMaxOutputTokens} |\n`);
    stream.markdown(`| enableToolCalling | ${cfg.enableToolCalling} |\n`);
    stream.markdown(`| parallelToolCalling | ${cfg.parallelToolCalling} |\n`);
    stream.markdown(`| agentTemperature | ${cfg.agentTemperature} |\n`);
    stream.markdown(`| streamingIdleTimeout | ${cfg.streamingIdleTimeout}ms |\n`);
    stream.markdown(`| maxRetries | ${cfg.maxRetries} |\n`);
    stream.markdown(`| contextWarningThreshold | ${cfg.contextWarningThreshold}% |\n`);
    stream.markdown(`| contextHardLimit | ${cfg.contextHardLimit}% |\n`);
    stream.markdown(`| maxMessageHistory | ${cfg.maxMessageHistory} |\n`);
    stream.markdown(`| activePreset | ${cfg.activePreset} |\n`);
    stream.markdown(`| showThinking | ${cfg.showThinking} |\n`);
}

function renderChangelog(stream: vscode.ChatResponseStream): void {
    stream.markdown(`## LLM Gateway — What's New\n\n`);

    stream.markdown(`### Sampling Presets\n`);
    stream.markdown(`- 12 task-specific presets: codegen, refactor, testgen, debug, review, docs, architect, cli, infra, explore, git, transform\n`);
    stream.markdown(`- File-based: edit JSON files in \`~/.config/llm-gateway/presets/\`, auto-reloads on save\n`);
    stream.markdown(`- Backend-aware: DRY, Mirostat, XTC, dynamic temp forwarded to llama.cpp; filtered for vLLM/OpenAI\n\n`);

    stream.markdown(`### Inline Overrides\n`);
    stream.markdown(`Type in any chat message:\n`);
    stream.markdown(`- \`/temp 0.0\` — one-shot temperature override\n`);
    stream.markdown(`- \`/preset cli\` — one-shot preset switch\n`);
    stream.markdown(`- \`/grammar root ::= "yes" | "no"\` — GBNF grammar constraint (llama.cpp)\n`);
    stream.markdown(`- \`/schema {"type":"object"}\` — JSON schema constraint\n\n`);

    stream.markdown(`### Thinking / Reasoning\n`);
    stream.markdown(`- \`showThinking\` setting: display model's reasoning in blockquote before response\n`);
    stream.markdown(`- \`reasoning_budget\` in presets: 0=disable, -1=infinite, >0=token limit\n\n`);

    stream.markdown(`### Status Bar\n`);
    stream.markdown(`- Live tokens/sec + context usage % after each request\n`);
    stream.markdown(`- Warning icon at 80%+ context usage\n\n`);

    stream.markdown(`### Commands\n`);
    stream.markdown(`- **Show Model Stats** — speed, context %, request count\n`);
    stream.markdown(`- **Export Conversation** — save chat to markdown file\n`);
    stream.markdown(`- **Select/Create/Open Presets** — manage sampling presets\n\n`);

    stream.markdown(`### Bug Fixes\n`);
    stream.markdown(`- Fixed llama.cpp tool calls not being detected (choice-level \`tool_calls\` format)\n`);
}

async function switchPreset(
    name: string,
    stream: vscode.ChatResponseStream,
    provider: GatewayProvider
): Promise<void> {
    if (!name) {
        stream.markdown(`Specify a preset name: \`@llm-gateway /preset codegen\`\n\n`);
        stream.markdown(`Available: ${Object.keys(provider.getPresets()).join(', ')}\n`);
        return;
    }

    const presets = provider.getPresets();
    if (!(name in presets)) {
        stream.markdown(`Unknown preset **${name}**. Available: ${Object.keys(presets).join(', ')}\n`);
        return;
    }

    await vscode.workspace.getConfiguration('github.copilot.llm-gateway')
        .update('activePreset', name, vscode.ConfigurationTarget.Global);

    const preset = presets[name];
    stream.markdown(`Switched to **${preset.name}** preset (temp: ${preset.params.temperature ?? '-'}, top_p: ${preset.params.top_p ?? '-'})\n`);
}

function renderHelp(stream: vscode.ChatResponseStream): void {
    stream.markdown(`## @llm-gateway\n\n`);
    stream.markdown(`Available commands:\n\n`);
    stream.markdown(`- \`/stats\` — Server stats, speed, context usage\n`);
    stream.markdown(`- \`/presets\` — List all presets with parameters\n`);
    stream.markdown(`- \`/preset <name>\` — Switch active preset\n`);
    stream.markdown(`- \`/config\` — Show current configuration\n`);
    stream.markdown(`- \`/changelog\` — What's new in this version\n`);
}
