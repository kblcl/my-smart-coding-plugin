const vscode = require('vscode');
const { getEditorContext } = require('./editorContext.js');
const { generateRecommendedCode } = require('./completionModule.js');

// 模块级变量：API Key 由 extension.js 注入
let API_KEY = null;
let completionTimeout = null;

/**
 * 设置 API Key（由 extension.js 调用）
 */
function setApiKey(key) {
    API_KEY = key;
}

/**
 * 创建自动补全 Provider
 */
function createInlineCompletionProvider() {
    return vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**/*' },
        {
            provideInlineCompletionItems(document, position) {
                return new Promise((resolve) => {
                    // 防抖：用户停顿时再触发
                    if (completionTimeout) clearTimeout(completionTimeout);

                    const prefix = document.lineAt(position.line).text.substring(0, position.character);
                    if (prefix.length < 2) {
                        return resolve({ items: [] });
                    }

                    completionTimeout = setTimeout(async () => {
                        try {
                            // 1. 获取上下文
                            const contextData = await getEditorContext({
                                mode: "partial",
                                beforeLines: 30,
                                afterLines: 10
                            });
                            if (contextData.error) {
                                console.warn('上下文获取失败:', contextData.error);
                                return resolve({ items: [] });
                            }

                            contextData.task = "补全光标位置的代码";

                            // 2. 调用 AI 生成补全
                            const result = await generateRecommendedCode(contextData, {
                                apiKey: API_KEY,
                                candidateCount: 1,
                                model: "qwen-turbo"
                            });
                            if (result.error || !result.candidates || result.candidates.length === 0) {
                                console.warn('AI补全失败:', result.error);
                                return resolve({ items: [] });
                            }

                            const suggestion = result.candidates[0].code;

                            // 3. 构造 InlineCompletionItem
                            // 构造 InlineCompletionItem（光标位置不变）
const item = new vscode.InlineCompletionItem(
    suggestion,
    new vscode.Range(position.line, position.character, position.line, position.character)
);

// ===== 新增：计算多行补全的实际范围 =====
const suggestionLines = suggestion.split('\n');
const endLine = position.line + suggestionLines.length - 1;
const endCharacter = (suggestionLines.length === 1) 
    ? position.character + suggestion.length 
    : suggestionLines[suggestionLines.length - 1].length;

const fullRange = new vscode.Range(
    position.line, position.character,
    endLine, endCharacter
);

// ===== 绑定高亮命令（使用 fullRange） =====
item.command = {
    command: 'my-smart-coding-plugin.highlightDiff',
    title: '高亮补全',
    arguments: [document, fullRange]
};

                            resolve({ items: [item] });

                        } catch (e) {
                            console.warn('自动补全出错:', e.message);
                            resolve({ items: [] });
                        }
                    }, 300);
                });
            }
        }
    );
}

/**
 * 创建高亮命令（D10）
 */
function createHighlightCommand() {
    return vscode.commands.registerCommand('my-smart-coding-plugin.highlightDiff', function (doc, range) {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        const decoration = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(0, 255, 0, 0.25)',
            border: '1px solid #00ff00'
        });
        editor.setDecorations(decoration, [range]);
        setTimeout(() => decoration.dispose(), 3000);
    });
}

module.exports = {
    setApiKey,
    createInlineCompletionProvider,
    createHighlightCommand
};