import * as vscode from 'vscode';
import { GatewayProvider } from './provider';

/**
 * Extension activation
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('GitHub Copilot LLM Gateway');
  outputChannel.appendLine('GitHub Copilot LLM Gateway extension is now active');

  const provider = new GatewayProvider(context, outputChannel);

  context.subscriptions.push(
    vscode.lm.registerLanguageModelChatProvider('copilot-llm-gateway', provider)
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      'github.copilot.llm-gateway.testConnection',
      async () => {
        try {
          const models = await provider.provideLanguageModelChatInformation(
            { silent: false },
            new vscode.CancellationTokenSource().token
          );

          if (models.length > 0) {
            vscode.window.showInformationMessage(
              `GitHub Copilot LLM Gateway: Successfully connected! Found ${models.length} model(s): ${models.map(m => m.name).join(', ')}`
            );
          } else {
            vscode.window.showWarningMessage(
              'GitHub Copilot LLM Gateway: Connected but no models found.'
            );
          }
        } catch (error) {
          vscode.window.showErrorMessage(
            `GitHub Copilot LLM Gateway: Connection test failed. ${error instanceof Error ? error.message : String(error)}`
          );
        }
      }
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      'github.copilot.llm-gateway.refreshModels',
      () => provider.refreshModels()
    )
  );

  context.subscriptions.push(outputChannel);

  outputChannel.appendLine('Copilot LLM Gateway provider registered with vendor ID: copilot-llm-gateway');
}

/**
 * Extension deactivation
 */
export function deactivate() {
  // Cleanup handled by context.subscriptions
}
