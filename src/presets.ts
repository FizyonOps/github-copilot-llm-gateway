/**
 * Sampling preset system for different backends and use cases.
 *
 * Presets define per-request sampling parameters that override server defaults.
 * Backend-aware: llama.cpp supports parameters (min_p, repeat_penalty, samplers, etc.)
 * that vLLM or generic OpenAI servers do not.
 */

export type BackendType = 'llamacpp' | 'vllm' | 'openai' | 'unknown';

export interface SamplingPreset {
    name: string;
    description: string;
    /** Which backends support this preset's extended params */
    backends: BackendType[];
    /** Sampling parameters sent per-request */
    params: SamplingParams;
}

export interface SamplingParams {
    // OpenAI-compatible (all backends)
    temperature?: number;
    top_p?: number;
    top_k?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    seed?: number;
    stop?: string[];

    // llama.cpp extended
    min_p?: number;
    repeat_penalty?: number;
    repeat_last_n?: number;
    typical_p?: number;
    mirostat?: number;
    mirostat_tau?: number;
    mirostat_eta?: number;
    dry_multiplier?: number;
    dry_base?: number;
    dry_allowed_length?: number;
    dry_penalty_last_n?: number;
    top_n_sigma?: number;
    xtc_probability?: number;
    xtc_threshold?: number;
    samplers?: string;
    dynatemp_range?: number;
    dynatemp_exponent?: number;
    reasoning_budget?: number;  // llama.cpp: 0=disable thinking, -1=infinite, >0=token limit

    // vLLM extended
    repetition_penalty?: number;
    length_penalty?: number;
    best_of?: number;
}

/** Keys valid for all OpenAI-compatible backends */
const OPENAI_KEYS: (keyof SamplingParams)[] = [
    'temperature', 'top_p', 'top_k', 'frequency_penalty',
    'presence_penalty', 'seed', 'stop',
];

/** Additional keys llama.cpp accepts per-request */
const LLAMACPP_KEYS: (keyof SamplingParams)[] = [
    ...OPENAI_KEYS,
    'min_p', 'repeat_penalty', 'repeat_last_n', 'typical_p',
    'mirostat', 'mirostat_tau', 'mirostat_eta',
    'dry_multiplier', 'dry_base', 'dry_allowed_length', 'dry_penalty_last_n',
    'top_n_sigma', 'xtc_probability', 'xtc_threshold',
    'samplers', 'dynatemp_range', 'dynatemp_exponent', 'reasoning_budget',
];

/** Additional keys vLLM accepts per-request */
const VLLM_KEYS: (keyof SamplingParams)[] = [
    ...OPENAI_KEYS,
    'repetition_penalty', 'length_penalty', 'best_of',
];

/**
 * Get the set of allowed parameter keys for a given backend.
 */
export function getAllowedKeysForBackend(backend: BackendType): string[] {
    switch (backend) {
        case 'llamacpp': return LLAMACPP_KEYS as string[];
        case 'vllm': return VLLM_KEYS as string[];
        default: return OPENAI_KEYS as string[];
    }
}

/**
 * Known llama.cpp chat template families, mirroring llama-chat.h llm_chat_template enum.
 * Only the subset relevant for message-format decisions is listed.
 */
export type LlamaChatTemplate =
    | 'chatml'        // generic: <|im_start|>role\ncontent<|im_end|>
    | 'llama2'        // only user/assistant/system, no tool role
    | 'llama3'        // <|start_header_id|>role — supports tool
    | 'llama4'        // <|header_start|>role   — supports tool
    | 'mistral'       // [INST]/[SYSTEM_PROMPT] variants, tool support varies
    | 'gemma'         // no system role; no tool
    | 'phi3' | 'phi4' // phi — no tool role
    | 'deepseek'      // deepseek v2/v3 — no tool
    | 'exaone'        // exaone 3 (no tool) / exaone 4 (tool yes)
    | 'kimi_k2'       // Kimi-K2 — supports tool
    | 'granite4'      // Granite 4.0 — supports tool
    | 'unknown';

/** Runtime capabilities derived from the detected chat template. */
export interface TemplateCaps {
    template: LlamaChatTemplate;
    /** Whether the 'tool' role message is correctly handled by the template */
    supportsToolRole: boolean;
    /** Whether a standalone 'system' role message is correctly handled */
    supportsSystemRole: boolean;
    /** Whether the template itself emits <think> tokens (native reasoning) */
    hasNativeThinking: boolean;
}

/**
 * Identify the llama.cpp chat template from its raw Jinja string.
 * Mirrors the heuristics in llama-chat.cpp (llama_chat_detect_template).
 * See: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp
 */
export function detectTemplateFromJinja(tmpl: string): TemplateCaps {
    const c = (s: string) => tmpl.includes(s);

    if (c('<|header_start|>') && c('<|header_end|>')) {
        return { template: 'llama4', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    if (c('<|start_header_id|>') && c('<|end_header_id|>')) {
        return { template: 'llama3', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    if (c('<|im_assistant|>assistant<|im_middle|>')) {
        return { template: 'kimi_k2', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    if (c('<|start_of_role|>') && (c('<tool_call>') || c('<tools>'))) {
        return { template: 'granite4', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    // EXAONE 4 has explicit [|tool|]
    if (c('[|system|]') && c('[|assistant|]') && c('[|endofturn|]') && c('[|tool|]')) {
        return { template: 'exaone', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    // EXAONE 3 — no tool
    if (c('[|system|]') && c('[|assistant|]') && c('[|endofturn|]')) {
        return { template: 'exaone', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: false };
    }
    // Gemma — no system, no tool
    if (c('<start_of_turn>')) {
        return { template: 'gemma', supportsToolRole: false, supportsSystemRole: false, hasNativeThinking: false };
    }
    // DeepSeek V3 (unicode fullwidth pipes)
    if (c('\uFF5CAssistant\uFF5C') && c('\uFF5CUser\uFF5C')) {
        return { template: 'deepseek', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: true };
    }
    // DeepSeek V2
    if (c("'Assistant: ' + message['content'] + eos_token")) {
        return { template: 'deepseek', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: false };
    }
    // Phi 4 — <|im_sep|> distinguishes from generic ChatML
    if (c('<|im_start|>') && c('<|im_sep|>')) {
        return { template: 'phi4', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: false };
    }
    // Mistral — [INST] based
    if (c('[INST]')) {
        return { template: 'mistral', supportsToolRole: false, supportsSystemRole: c('[SYSTEM_PROMPT]'), hasNativeThinking: false };
    }
    // Llama 2
    if (c('<<SYS>>')) {
        return { template: 'llama2', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: false };
    }
    // ChatML — generic role passthrough; 'tool' role becomes <|im_start|>tool\n...<|im_end|>
    if (c('<|im_start|>')) {
        return { template: 'chatml', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
    }
    // Phi 3 uses <|role|> style but no <|im_start|>
    if (c('<|end|>')) {
        return { template: 'phi3', supportsToolRole: false, supportsSystemRole: true, hasNativeThinking: false };
    }
    return { template: 'unknown', supportsToolRole: true, supportsSystemRole: true, hasNativeThinking: false };
}

/**
 * Detect backend type from the /v1/models response.
 */
export function detectBackend(modelsResponse: { data: Array<{ owned_by?: string }> }): BackendType {
    if (!modelsResponse.data || modelsResponse.data.length === 0) return 'unknown';
    const owner = modelsResponse.data[0].owned_by?.toLowerCase() ?? '';
    if (owner.includes('llamacpp') || owner.includes('llama.cpp') || owner.includes('llama-cpp')) return 'llamacpp';
    if (owner.includes('vllm')) return 'vllm';
    if (owner.includes('openai')) return 'openai';
    return 'unknown';
}

/**
 * Built-in presets. Users can override/extend via settings.
 *
 * Design principles:
 * - Each preset targets a SPECIFIC development task, not just a temperature knob
 * - llama.cpp extended params (DRY, Mirostat, XTC, samplers order) are tuned per scenario
 * - Sampler ordering matters: penalties→dry→top_k→top_p→min_p→temperature is precision-first
 */
export const BUILTIN_PRESETS: Record<string, SamplingPreset> = {

    // ── Code Generation ────────────────────────────────────────────────
    codegen: {
        name: 'Code Gen',
        description: 'Writing new functions, components, endpoints. Precise but not robotic — allows minor variance for natural code style',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.2,
            top_p: 0.9,
            top_k: 20,
            min_p: 0.05,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,  // code repetition is intentional (loops, patterns)
        },
    },

    // ── Refactoring & Formatting ───────────────────────────────────────
    refactor: {
        name: 'Refactor',
        description: 'Deterministic transforms — rename, extract, restructure. Reproducible output, zero creativity',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            seed: 42,
        },
    },

    // ── Test Generation ────────────────────────────────────────────────
    testgen: {
        name: 'Test Gen',
        description: 'Unit/integration tests. Slightly creative for edge cases, but structured output. DRY penalty avoids copy-paste test bodies',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.35,
            top_p: 0.92,
            top_k: 30,
            min_p: 0.04,
            frequency_penalty: 0.05,  // slight nudge to vary test names/descriptions
            repeat_penalty: 1.05,     // light — test structure is intentionally repetitive
            repeat_last_n: 128,
            dry_multiplier: 0.5,      // penalize verbatim repeated test bodies
            dry_base: 1.75,
            dry_allowed_length: 4,
        },
    },

    // ── Debugging & Root Cause Analysis ────────────────────────────────
    debug: {
        name: 'Debug',
        description: 'Stack trace analysis, root cause finding. Mirostat sampling for consistent perplexity — methodical step-by-step reasoning without wandering',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.25,
            top_p: 0.9,
            top_k: 30,
            min_p: 0.04,
            repeat_penalty: 1.08,    // reasoning chains can loop on the same phrasing
            repeat_last_n: 128,
            mirostat: 2,
            mirostat_tau: 4.0,   // lower tau = more focused/less creative
            mirostat_eta: 0.1,
        },
    },

    // ── Code Review & Security Audit ───────────────────────────────────
    review: {
        name: 'Review',
        description: 'Code review, bug hunting, security audit. Thorough coverage — presence penalty forces exploring different angles instead of repeating the same finding',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.3,
            top_p: 0.92,
            top_k: 40,
            min_p: 0.03,
            presence_penalty: 0.3,   // don't repeat the same issue/pattern
            frequency_penalty: 0.1,  // vary vocabulary for different findings
            repeat_penalty: 1.1,     // avoid restating the same finding
            repeat_last_n: 256,
        },
    },

    // ── Documentation & Comments ───────────────────────────────────────
    docs: {
        name: 'Docs',
        description: 'JSDoc, README, API docs, comments. Natural prose with DRY penalty against boilerplate repetition across docstrings',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.5,
            top_p: 0.93,
            top_k: 50,
            min_p: 0.02,
            presence_penalty: 0.2,  // vary phrasing across doc blocks
            repeat_penalty: 1.1,    // avoid "This function... This function..." loops
            repeat_last_n: 256,
            dry_multiplier: 0.8,    // penalize boilerplate phrases ("This function returns...")
            dry_base: 1.75,
            dry_allowed_length: 3,
        },
    },

    // ── Architecture & Design Decisions ────────────────────────────────
    architect: {
        name: 'Architect',
        description: 'System design, API design, tradeoff analysis. Higher creativity to explore alternatives, presence penalty to cover multiple angles',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.6,
            top_p: 0.94,
            top_k: 50,
            min_p: 0.02,
            presence_penalty: 0.4,   // force exploring different design options
            frequency_penalty: 0.15, // varied vocabulary for pros/cons
            repeat_penalty: 1.1,     // prevent circling back to same arguments
            repeat_last_n: 256,
        },
    },

    // ── Shell / CLI Commands ───────────────────────────────────────────
    cli: {
        name: 'CLI',
        description: 'Shell commands, one-liners, scripts. Maximum precision — a wrong flag can be destructive. No creativity, no variance',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        },
    },

    // ── Infrastructure (K8s, Docker, Terraform) ────────────────────────
    infra: {
        name: 'Infra',
        description: 'K8s manifests, Dockerfiles, Terraform, CI configs. Low temp for correctness but slight variance for naming. XTC smooths token distribution',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.15,
            top_p: 0.9,
            top_k: 20,
            min_p: 0.05,
            xtc_probability: 0.1,  // smooth out spiky logits in YAML/config tokens
            xtc_threshold: 0.15,
        },
    },

    // ── Brainstorm & Exploration ───────────────────────────────────────
    explore: {
        name: 'Explore',
        description: 'Brainstorming, "what if", idea generation, learning new concepts. Dynamic temperature widens the beam adaptively',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 80,
            min_p: 0.01,
            presence_penalty: 0.3,
            frequency_penalty: 0.15,
            repeat_penalty: 1.1,     // brainstorming can loop on the same ideas
            repeat_last_n: 256,
            dynatemp_range: 0.3,     // temp varies 0.5–1.1 adaptively
            dynatemp_exponent: 1.0,
        },
    },

    // ── Git (Commits, PR, Changelog) ───────────────────────────────────
    git: {
        name: 'Git',
        description: 'Commit messages, PR descriptions, changelogs. Concise, consistent formatting. DRY prevents "Updated X. Updated Y. Updated Z." patterns',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.3,
            top_p: 0.9,
            top_k: 30,
            min_p: 0.04,
            repeat_penalty: 1.1,    // changelog entries love repeating verbs
            repeat_last_n: 256,
            dry_multiplier: 1.0,    // strongly penalize repeated sentence starters
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 256,
        },
    },

    // ── Data Transform (JSON, YAML, CSV, Regex) ────────────────────────
    transform: {
        name: 'Transform',
        description: 'JSON/YAML manipulation, regex, data conversion. Deterministic with seed for reproducibility across runs',
        backends: ['llamacpp', 'vllm', 'openai', 'unknown'],
        params: {
            temperature: 0.05,
            top_p: 0.95,
            top_k: 10,
            min_p: 0.08,
            seed: 42,
        },
    },
};

/**
 * Filter preset params to only include keys the backend understands.
 */
export function filterParamsForBackend(params: SamplingParams, backend: BackendType): Record<string, unknown> {
    const allowed = new Set(getAllowedKeysForBackend(backend));
    const filtered: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && allowed.has(key)) {
            filtered[key] = value;
        }
    }
    return filtered;
}

// ── File-based Preset Management ─────────────────────────────────────

import * as fs from 'node:fs';
import * as path from 'node:path';

const DEFAULT_PRESETS_DIR = path.join(
    process.env.HOME || process.env.USERPROFILE || '~',
    '.config', 'llm-gateway', 'presets'
);

/**
 * Resolve the presets directory. Uses configured path or default.
 */
export function resolvePresetsDir(configuredPath?: string): string {
    if (configuredPath && configuredPath.trim()) {
        // Expand ~ to home directory
        let resolved = configuredPath.trim();
        if (resolved.startsWith('~/') || resolved === '~') {
            const home = process.env.HOME || process.env.USERPROFILE || '';
            resolved = path.join(home, resolved.slice(1));
        }
        return path.resolve(resolved);
    }
    return DEFAULT_PRESETS_DIR;
}

/**
 * Ensure presets directory exists. If empty, seed with built-in presets.
 * Returns the resolved directory path.
 */
export function ensurePresetsDir(presetsDir: string): string {
    if (!fs.existsSync(presetsDir)) {
        fs.mkdirSync(presetsDir, { recursive: true });
    }

    // Seed built-in presets if directory has no .json files
    const existingFiles = fs.readdirSync(presetsDir).filter(f => f.endsWith('.json'));
    if (existingFiles.length === 0) {
        seedBuiltinPresets(presetsDir);
    }

    return presetsDir;
}

/**
 * Write all built-in presets as individual JSON files.
 */
function seedBuiltinPresets(presetsDir: string): void {
    for (const [key, preset] of Object.entries(BUILTIN_PRESETS)) {
        const filePath = path.join(presetsDir, `${key}.json`);
        const content = JSON.stringify(preset, null, 2) + '\n';
        fs.writeFileSync(filePath, content, 'utf-8');
    }
}

/**
 * Load all presets from JSON files in the presets directory.
 * Each file should contain a single SamplingPreset object.
 * File name (without .json) becomes the preset key.
 */
export function loadPresetsFromDir(presetsDir: string, log?: (msg: string) => void): Record<string, SamplingPreset> {
    const presets: Record<string, SamplingPreset> = {};

    if (!fs.existsSync(presetsDir)) return presets;

    const files = fs.readdirSync(presetsDir).filter(f => f.endsWith('.json'));

    for (const file of files) {
        const key = path.basename(file, '.json');
        const filePath = path.join(presetsDir, file);
        try {
            const raw = fs.readFileSync(filePath, 'utf-8');
            const preset = JSON.parse(raw) as SamplingPreset;

            // Basic validation
            if (!preset.name || !preset.params || typeof preset.params !== 'object') {
                log?.(`WARNING: Skipping invalid preset file ${file} (missing name or params)`);
                continue;
            }
            if (!preset.backends) {
                preset.backends = ['llamacpp', 'vllm', 'openai', 'unknown'];
            }
            if (!preset.description) {
                preset.description = '';
            }

            presets[key] = preset;
        } catch (err) {
            log?.(`WARNING: Failed to load preset ${file}: ${err instanceof Error ? err.message : String(err)}`);
        }
    }

    return presets;
}

/**
 * Save a single preset to a JSON file.
 */
export function savePresetToFile(presetsDir: string, key: string, preset: SamplingPreset): void {
    if (!fs.existsSync(presetsDir)) {
        fs.mkdirSync(presetsDir, { recursive: true });
    }
    const filePath = path.join(presetsDir, `${key}.json`);
    const content = JSON.stringify(preset, null, 2) + '\n';
    fs.writeFileSync(filePath, content, 'utf-8');
}

/**
 * Get the file path for a preset key.
 */
export function getPresetFilePath(presetsDir: string, key: string): string {
    return path.join(presetsDir, `${key}.json`);
}
