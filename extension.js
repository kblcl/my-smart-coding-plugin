const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const axios = require('axios');

// 引入拆分后的模块
const { setApiKey, createInlineCompletionProvider, createHighlightCommand } = require('./inlineCompletionProvider.js');

// 导入队友的核心模块
const { getEditorContext } = require('./editorContext.js');
const { generateRecommendedCode } = require('./completionModule.js');

const BASE_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation';
const MODEL_NAME = 'qwen-turbo';

// 模块级变量
let API_KEY = null;

function activate(context) {
    // ========== 加载 .env ==========
    const envPath = path.join(context.extensionPath, '.env');
    console.log('📂 读取 .env 路径:', envPath);

    try {
        if (fs.existsSync(envPath)) {
            const envContent = fs.readFileSync(envPath, 'utf-8');
            const match = envContent.match(/AI_API_KEY=(.+)/);
            if (match) {
                API_KEY = match[1].trim();
                console.log('✅ 从 .env 读取 Key 成功:', API_KEY.slice(0, 10) + '...');
            } else {
                console.warn('⚠️ .env 中未找到 AI_API_KEY');
            }
        } else {
            console.warn('⚠️ .env 文件不存在');
        }
    } catch (err) {
        console.warn('⚠️ 读取 .env 失败:', err.message);
    }

    // // 兜底：硬编码（仅限本地调试）
    // if (!API_KEY) {
    //     API_KEY = '你的密钥';
    //     console.warn('⚠️ 使用硬编码 API Key（请勿提交到 GitHub）');
    // }
// 如果未加载成功，报错提示用户配置 .env
if (!API_KEY) {
    console.error('❌ API Key 加载失败，请检查 .env 文件');
    vscode.window.showErrorMessage('API Key 未配置，请创建 .env 文件并设置 AI_API_KEY');
    return; // 停止激活插件
}
    // 将 API Key 注入到 inlineCompletionProvider 模块
    setApiKey(API_KEY);

    console.log('🔑 API Key 状态:', API_KEY ? '已加载' : '未加载');

    // ---------- 功能1：插入 Console Log ----------
    let insertLog = vscode.commands.registerCommand('my-smart-coding-plugin.insertLog', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        const selectedText = editor.document.getText(editor.selection);
        if (!selectedText) {
            vscode.window.showWarningMessage('请先选中一个变量名！');
            return;
        }
        editor.edit(editBuilder => {
            editBuilder.replace(editor.selection, `console.log('${selectedText}:', ${selectedText});`);
        });
    });

    // ---------- 功能2：AI 解释代码 ----------
    let explainCode = vscode.commands.registerCommand('my-smart-coding-plugin.explainCode', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        const selectedCode = editor.document.getText(editor.selection);
        if (!selectedCode) {
            vscode.window.showWarningMessage('请先选中一段代码！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('API Key 未配置');
            return;
        }
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "AI 正在分析代码..."
        }, async () => {
            try {
                const response = await axios.post(BASE_URL, {
                    model: MODEL_NAME,
                    input: { messages: [{ role: 'user', content: `解释这段代码：\n${selectedCode}` }] },
                    parameters: { result_format: 'message' }
                }, {
                    headers: { 'Authorization': `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
                    timeout: 15000
                });
                const answer = response.data.output.choices[0].message.content;
                vscode.window.showInformationMessage(answer, { modal: true });
            } catch (error) {
                vscode.window.showErrorMessage('AI调用失败: ' + (error.response?.data?.message || error.message));
            }
        });
    });

    // ---------- 功能3：手动智能补全 ----------
    let smartComplete = vscode.commands.registerCommand('my-smart-coding-plugin.smartComplete', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个代码文件！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('API Key 未配置');
            return;
        }

        let contextData;
        try {
            contextData = await getEditorContext({ mode: "partial", beforeLines: 50, afterLines: 50 });
        } catch (err) {
            vscode.window.showErrorMessage('上下文获取失败：' + err.message);
            return;
        }
        if (contextData.error) {
            vscode.window.showErrorMessage('上下文错误：' + contextData.error);
            return;
        }

        const task = await vscode.window.showInputBox({
            prompt: '请输入你想让 AI 做什么',
            placeHolder: '例如：补全函数、生成注释',
            value: '补全当前代码'
        });
        if (!task) return;
        contextData.task = task;

        let aiResult;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "AI 正在生成代码..."
        }, async () => {
            try {
                aiResult = await generateRecommendedCode(contextData, {
                    apiKey: API_KEY,
                    candidateCount: 3,
                    model: "qwen-turbo"
                });
            } catch (err) {
                vscode.window.showErrorMessage('AI 调用失败：' + err.message);
            }
        });

        if (!aiResult || aiResult.error) {
            vscode.window.showErrorMessage('AI生成失败：' + (aiResult?.error || '未知错误'));
            return;
        }

        const quickPickItems = aiResult.candidates.map(candidate => ({
            label: `候选 ${candidate.id}`,
            description: candidate.description,
            detail: candidate.code.slice(0, 100)
        }));

        const selected = await vscode.window.showQuickPick(quickPickItems, {
            placeHolder: '请选择要插入的代码候选'
        });

        if (selected) {
            const selectedCandidate = aiResult.candidates.find(c => `候选 ${c.id}` === selected.label);
            if (selectedCandidate) {
                editor.edit(editBuilder => {
                    editBuilder.insert(editor.selection.active, selectedCandidate.code);
                });
                vscode.window.showInformationMessage('✅ 代码已插入！');
            }
        }
    });

    // ---------- 功能4：一键测试完整流程 ----------
    let testFullFlow = vscode.commands.registerCommand('my-smart-coding-plugin.testFullFlow', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个代码文件！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('API Key 未配置');
            return;
        }

        console.log('🧪 开始完整流程测试...');
        let contextData;
        try {
            contextData = await getEditorContext({ mode: "partial", beforeLines: 50, afterLines: 50 });
            console.log('✅ 上下文获取成功');
        } catch (err) {
            vscode.window.showErrorMessage('第一步失败：' + err.message);
            return;
        }
        if (contextData.error) {
            vscode.window.showErrorMessage('上下文错误：' + contextData.error);
            return;
        }

        const task = await vscode.window.showInputBox({
            prompt: '确认AI任务',
            value: '补全当前光标位置的代码，保持和前后风格一致'
        });
        if (!task) return;
        contextData.task = task;

        let aiResult;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "正在调用AI..."
        }, async () => {
            try {
                aiResult = await generateRecommendedCode(contextData, {
                    apiKey: API_KEY,
                    candidateCount: 2,
                    model: "qwen-plus"
                });
                console.log('✅ AI生成成功');
            } catch (err) {
                vscode.window.showErrorMessage('第二步失败：' + err.message);
            }
        });

        if (!aiResult || aiResult.error) {
            vscode.window.showErrorMessage('AI生成失败：' + (aiResult?.error || '未知错误'));
            return;
        }

        const selected = await vscode.window.showQuickPick(
            aiResult.candidates.map(c => ({
                label: `候选 ${c.id}`,
                description: c.description,
                detail: c.code
            })),
            { placeHolder: '选择要插入的代码' }
        );

        if (selected) {
            const code = aiResult.candidates.find(c => `候选 ${c.id}` === selected.label)?.code;
            if (code) {
                editor.edit(eb => eb.insert(editor.selection.active, code));
                vscode.window.showInformationMessage('✅ 代码已插入！');
            }
        }
        console.log('🎉 完整流程测试结束！');
    });

    // ================================================================
    // 从 inlineCompletionProvider.js 创建并注册 D6 和 D10
    // ================================================================
    const inlineProvider = createInlineCompletionProvider();
    const highlightCmd = createHighlightCommand();

    // ---------- 注册所有功能 ----------
    context.subscriptions.push(
        insertLog,
        explainCode,
        smartComplete,
        testFullFlow,
        inlineProvider,
        highlightCmd
    );

    vscode.window.showInformationMessage('✅ HIT 智能编码助手已加载（含自动补全）');
    console.log('✅ 所有功能注册完成');
}

function deactivate() {}

module.exports = { activate, deactivate };