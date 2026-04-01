import * as vscode from 'vscode';

export interface WorkspaceInstructionFile {
    readonly uri: vscode.Uri;
    readonly label: string;
    readonly content: string;
}

const ALWAYS_ON_WORKSPACE_INSTRUCTION_PATHS = ['.github/copilot-instructions.md'];
const AGENTS_INSTRUCTION_PATHS = ['AGENTS.md'];
const CLAUDE_INSTRUCTION_PATHS = [
    'CLAUDE.md',
    'CLAUDE.local.md',
    '.claude/CLAUDE.md',
    '.claude/CLAUDE.local.md',
];

async function readTextFileIfPresent(uri: vscode.Uri): Promise<string | undefined> {
    try {
        const bytes = await vscode.workspace.fs.readFile(uri);
        const content = new TextDecoder().decode(bytes).trim();
        return content || undefined;
    } catch {
        return undefined;
    }
}

function buildCandidateUris(): vscode.Uri[] {
    const folders = vscode.workspace.workspaceFolders ?? [];
    const chatConfig = vscode.workspace.getConfiguration('chat');
    const useAgentsMdFile = chatConfig.get<boolean>('useAgentsMdFile', true);
    const useClaudeMdFile = chatConfig.get<boolean>('useClaudeMdFile', true);

    const relativePaths = [...ALWAYS_ON_WORKSPACE_INSTRUCTION_PATHS];
    if (useAgentsMdFile) {
        relativePaths.push(...AGENTS_INSTRUCTION_PATHS);
    }
    if (useClaudeMdFile) {
        relativePaths.push(...CLAUDE_INSTRUCTION_PATHS);
    }

    return folders.flatMap((folder) => relativePaths.map((relativePath) => vscode.Uri.joinPath(folder.uri, relativePath)));
}

export async function collectWorkspaceInstructionFiles(): Promise<WorkspaceInstructionFile[]> {
    const candidates = buildCandidateUris();
    const files = await Promise.all(
        candidates.map(async (uri) => {
            const content = await readTextFileIfPresent(uri);
            if (!content) {
                return undefined;
            }

            return {
                uri,
                label: vscode.workspace.asRelativePath(uri, true),
                content,
            } satisfies WorkspaceInstructionFile;
        })
    );

    return files
        .filter((file): file is WorkspaceInstructionFile => Boolean(file))
        .sort((left, right) => left.label.localeCompare(right.label));
}