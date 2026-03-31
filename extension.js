const vscode = require('vscode');
const path = require('path');

// 1. 严格按协议引入两个核心模块
const { getEditorContext } = require('./editorContext.js');
const { generateRecommendedCode } = require('./completionModule.js');

function activate(context) {
    // 加载环境变量（.env 里的 AI_API_KEY）
    require('dotenv').config({ path: path.join(context.extensionPath, '.env') });
    const API_KEY = process.env.AI_API_KEY;
    console.log('✅ 插件激活成功');
    console.log('API Key 状态:', API_KEY ? '已加载' : '未加载');

    // -----------------------------------------------------------------------
    // 原有功能1：插入 Console Log（保留）
    // -----------------------------------------------------------------------
    let insertLog = vscode.commands.registerCommand('my-smart-coding-plugin.insertLog', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return; }
        const selectedText = editor.document.getText(editor.selection);
        if (!selectedText) {
            vscode.window.showWarningMessage('请先选中一个变量名！');
            return;
        }
        const logStatement = `console.log('${selectedText}:', ${selectedText});`;
        editor.edit(editBuilder => {
            editBuilder.replace(editor.selection, logStatement);
        });
    });

    // -----------------------------------------------------------------------
    // 原有功能2：AI 解释代码（保留）
    // -----------------------------------------------------------------------
    let explainCode = vscode.commands.registerCommand('my-smart-coding-plugin.explainCode', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return; }
        const selectedCode = editor.document.getText(editor.selection);
        if (!selectedCode) {
            vscode.window.showWarningMessage('请先选中一段代码！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('请先在 .env 文件中配置 AI_API_KEY');
            return;
        }
        // （这里可以保留你原来的通义千问解释代码逻辑，或者也用 completionModule 改造）
        vscode.window.showInformationMessage('解释代码功能待集成 completionModule');
    });

    // -----------------------------------------------------------------------
    // 新增核心功能：AI 智能代码补全（上下文 + AI 结合）
    // -----------------------------------------------------------------------
    let smartComplete = vscode.commands.registerCommand('my-smart-coding-plugin.smartComplete', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个代码文件！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('请先在 .env 文件中配置 AI_API_KEY');
            return;
        }

        // ==========================================
        // 第一步：调用 getEditorContext 获取上下文（严格按协议）
        // ==========================================
        let contextData;
        try {
            contextData = await getEditorContext({
                mode: "partial", // 优先识别函数/类，识别不到取前后50行
                beforeLines: 50,
                afterLines: 50
            });
            console.log('✅ 上下文获取成功:', contextData);
        } catch (err) {
            vscode.window.showErrorMessage('上下文获取失败：' + err.message);
            return;
        }

        // 透传上下文模块的错误
        if (contextData.error && contextData.error.trim() !== "") {
            vscode.window.showErrorMessage('上下文错误：' + contextData.error);
            return;
        }

        // ==========================================
        // 第二步：让用户选择/输入补全任务（可选，提升体验）
        // ==========================================
        const task = await vscode.window.showInputBox({
            prompt: '请输入你想让 AI 做什么（例如：补全函数、生成注释、优化代码）',
            placeHolder: '例如：补全一个防抖函数',
            value: '补全当前代码'
        });
        if (!task) { return; } // 用户取消输入

        // 把任务加到 contextData 里（符合 completionModule 的入参要求）
        contextData.task = task;

        // ==========================================
        // 第三步：调用 generateRecommendedCode 生成候选代码
        // ==========================================
        let aiResult;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "AI 正在生成代码..."
        }, async () => {
            try {
                aiResult = await generateRecommendedCode(contextData, {
                    apiKey: API_KEY,
                    candidateCount: 3, // 生成3个候选让用户选
                    model: "qwen-turbo"
                });
            } catch (err) {
                vscode.window.showErrorMessage('AI 调用失败：' + err.message);
            }
        });

        if (!aiResult) { return; }
        if (aiResult.error && aiResult.error.trim() !== "") {
            vscode.window.showErrorMessage(aiResult.error);
            return;
        }
        if (!aiResult.candidates || aiResult.candidates.length === 0) {
            vscode.window.showWarningMessage('AI 没有生成任何候选代码');
            return;
        }

        console.log('✅ AI 生成成功:', aiResult);

        // ==========================================
        // 第四步：展示候选给用户选择，插入编辑器
        // ==========================================
        const quickPickItems = aiResult.candidates.map(candidate => ({
            label: `候选 ${candidate.id}`,
            description: candidate.description,
            detail: candidate.code.slice(0, 100) // 预览前100个字符
        }));

        const selected = await vscode.window.showQuickPick(quickPickItems, {
            placeHolder: '请选择要插入的代码候选',
            matchOnDescription: true,
            matchOnDetail: true
        });

        if (!selected) { return; } // 用户取消选择

        // 找到用户选的候选代码
        const selectedCandidate = aiResult.candidates.find(c => `候选 ${c.id}` === selected.label);
        if (!selectedCandidate || !selectedCandidate.code) {
            vscode.window.showWarningMessage('候选代码无效');
            return;
        }

        // ==========================================
        // 第五步：把代码插入到编辑器光标位置
        // ==========================================
        editor.edit(editBuilder => {
            const position = editor.selection.active;
            editBuilder.insert(position, selectedCandidate.code);
        });

        vscode.window.showInformationMessage('代码已插入！');
    });
// -----------------------------------------------------------------------
// 一键测试：上下文获取 + AI生成 完整流程
// -----------------------------------------------------------------------
    let testFullFlow = vscode.commands.registerCommand('my-smart-coding-plugin.testFullFlow', async function () {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('请先打开一个代码文件！');
        return;
    }
    const API_KEY = process.env.AI_API_KEY;
    if (!API_KEY) {
        vscode.window.showErrorMessage('请先在 .env 文件里配置 AI_API_KEY');
        return;
    }

    console.log('🧪 开始完整流程测试...');

    // ==========================================
    // 第一步：真实获取编辑器上下文
    // ==========================================
    let contextData;
    try {
        contextData = await getEditorContext({
        mode: "partial",
        beforeLines: 50,
        afterLines: 50
        });
        console.log('✅ 第一步完成：上下文获取成功');
        console.log('📄 上下文数据：', JSON.stringify(contextData, null, 2));
    } catch (err) {
        vscode.window.showErrorMessage('第一步失败：' + err.message);
        return;
    }
    if (contextData.error) {
        vscode.window.showErrorMessage('上下文错误：' + contextData.error);
        return;
    }

    // ==========================================
    // 第二步：让用户确认任务
    // ==========================================
    const task = await vscode.window.showInputBox({
        prompt: '确认AI任务（会结合当前光标上下文）',
        value: '补全当前光标位置的代码，保持和前后风格一致'
    });
    if (!task) return;
    contextData.task = task;

    // ==========================================
    // 第三步：调用AI生成
    // ==========================================
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
        console.log('✅ 第二步完成：AI生成成功');
        console.log('🤖 AI返回数据：', JSON.stringify(aiResult, null, 2));
        } catch (err) {
        vscode.window.showErrorMessage('第二步失败：' + err.message);
        }
    });

    if (!aiResult || aiResult.error) {
        vscode.window.showErrorMessage('AI生成失败：' + (aiResult?.error || '未知错误'));
        return;
    }

    // ==========================================
    // 第四步：展示结果并插入
    // ==========================================
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

    // 记得把 testFullFlow 也 push 进去
    context.subscriptions.push(insertLog, explainCode, smartComplete, testFullFlow);
    // 注册所有命令
    context.subscriptions.push(insertLog, explainCode, smartComplete);
}

function deactivate() {}

module.exports = { activate, deactivate };