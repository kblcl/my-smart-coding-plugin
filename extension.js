const vscode = require('vscode');
const path = require('path');

// 1. 严格按协议引入两个核心模块
const { getEditorContext } = require('./editorContext.js');
const { generateRecommendedCode } = require('./completionModule.js');

// 2. 引入郭鑫龙实现的自动补全模块
const { setApiKey, createInlineCompletionProvider, createHighlightCommand } = require('./inlineCompletionProvider.js');

// 3. 引入张瑞泽实现的教学引导模块
const { generateTeachingGuide, generateCodeExplanation } = require('./teaching/teachingMode.js');

function activate(context) {
    // 加载环境变量（.env 里的 AI_API_KEY）
    require('dotenv').config({ path: path.join(context.extensionPath, '.env') });
    const API_KEY = process.env.AI_API_KEY;
    console.log('✅ 插件激活成功');
    console.log('API Key 状态:', API_KEY ? '已加载' : '未加载');

    if (!API_KEY) {
        console.error('❌ API Key 加载失败，请检查 .env 文件');
        vscode.window.showErrorMessage('API Key 未配置，请创建 .env 文件并设置 AI_API_KEY');
        return;
    }

    // 将 API Key 注入到 inlineCompletionProvider 模块
    setApiKey(API_KEY);

    // 读取教学引导模式配置
    function getCompletionMode() {
        const config = vscode.workspace.getConfiguration();
        return config.get('hitsmartcoder.completionMode', 'standard');
    }

    // 读取模型后端配置（cloud=云端, local=本地微调模型）
    function getModelBackend() {
        const config = vscode.workspace.getConfiguration();
        return config.get('hitsmartcoder.modelBackend', 'cloud');
    }

    // 读取本地模型服务地址
    function getLocalEndpoint() {
        const config = vscode.workspace.getConfiguration();
        return config.get('hitsmartcoder.localEndpoint', 'http://localhost:8000');
    }

    // -----------------------------------------------------------------------
    // 功能1：插入 Console Log
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
    // 功能2：AI 解释代码（已集成至教学引导模块）
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

        const mode = getCompletionMode();
        const isTeachingMode = mode === 'explain' || mode === 'hint';

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: isTeachingMode ? "AI 正在教学式讲解代码..." : "AI 正在分析代码..."
        }, async () => {
            try {
                const result = await generateCodeExplanation(
                    selectedCode,
                    editor.document.languageId,
                    {
                        apiKey: API_KEY,
                        mode: isTeachingMode ? 'explain' : 'standard',
                        model: 'qwen-max'
                    }
                );

                if (result.error) {
                    vscode.window.showErrorMessage('解释失败：' + result.error);
                    return;
                }

                if (isTeachingMode) {
                    // 教学模式：用 Webview 弹窗展示（含知识点和思考题）
                    const panel = vscode.window.createWebviewPanel(
                        'codeExplain',
                        '代码讲解',
                        vscode.ViewColumn.Beside,
                        {}
                    );
                    panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; line-height: 1.6; }
h2 { color: #004DA0; border-bottom: 2px solid #004DA0; padding-bottom: 8px; }
.concept { background: #E8F0FE; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
.question { background: #FFF3E0; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
.explain { font-size: 15px; }
.tag { display: inline-block; background: #004DA0; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
</style></head><body>
<h2>📖 代码讲解 <span class="tag">教学</span></h2>
<div class="explain">${result.explanation.replace(/\n/g, '<br>')}</div>
${result.concepts.length ? `<div class="concept"><b>📌 涉及知识点</b><br>${result.concepts.map(c => '• ' + c).join('<br>')}</div>` : ''}
${result.questions.length ? `<div class="question"><b>🤔 延伸思考</b><br>${result.questions.map(q => '• ' + q).join('<br>')}</div>` : ''}
</body></html>`;
                } else {
                    // 标准模式：直接弹窗显示
                    vscode.window.showInformationMessage(result.explanation, { modal: true });
                }
            } catch (error) {
                vscode.window.showErrorMessage('AI调用失败: ' + (error.message || '未知错误'));
            }
        });
    });

    // -----------------------------------------------------------------------
    // 功能3：手动智能补全（多候选选择）— 根据 completionMode 切换行为
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

        const mode = getCompletionMode();

        // 教学模式下：重定向到教学引导逻辑
        if (mode !== 'standard') {
            // 复用 teachingGuide 的逻辑
            const modeNames = { hint: '提示引导', scaffold: '框架引导', explain: '讲解引导' };
            const tip = await vscode.window.showInformationMessage(
                `当前是「${modeNames[mode] || mode}」模式，要使用教学引导功能吗？`,
                `是，打开教学引导`,
                `切换为标准模式再补全`,
                '取消'
            );
            if (tip === `是，打开教学引导`) {
                // 触发教学引导命令
                vscode.commands.executeCommand('my-smart-coding-plugin.teachingGuide');
            } else if (tip === `切换为标准模式再补全`) {
                await vscode.workspace.getConfiguration().update('hitsmartcoder.completionMode', 'standard', true);
                vscode.window.showInformationMessage('已切换为标准补全模式，请再次运行补全命令');
            }
            return;
        }

        // ======== 标准模式：原有逻辑 ========
        let contextData;
        try {
            contextData = await getEditorContext({
                mode: "partial",
                beforeLines: 50,
                afterLines: 50
            });
        } catch (err) {
            vscode.window.showErrorMessage('上下文获取失败：' + err.message);
            return;
        }
        if (contextData.error && contextData.error.trim() !== "") {
            vscode.window.showErrorMessage('上下文错误：' + contextData.error);
            return;
        }

        const task = await vscode.window.showInputBox({
            prompt: '请输入你想让 AI 做什么',
            placeHolder: '例如：补全函数、生成注释、优化代码',
            value: '补全当前代码'
        });
        if (!task) { return; }
        contextData.task = task;

        let aiResult;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "AI 正在生成代码..."
        }, async () => {
            try {
                const backend = getModelBackend();
                const aiConfig = {
                    apiKey: API_KEY,
                    candidateCount: 3,
                    model: backend === 'local' ? 'qwen-coder-l2' : "qwen-turbo"
                };
                if (backend === 'local') {
                    aiConfig.endpoint = getLocalEndpoint();
                }
                aiResult = await generateRecommendedCode(contextData, aiConfig);
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

        const quickPickItems = aiResult.candidates.map(candidate => ({
            label: `候选 ${candidate.id}`,
            description: candidate.description,
            detail: candidate.code.slice(0, 100)
        }));

        const selected = await vscode.window.showQuickPick(quickPickItems, {
            placeHolder: '请选择要插入的代码候选',
            matchOnDescription: true,
            matchOnDetail: true
        });

        if (!selected) { return; }

        const selectedCandidate = aiResult.candidates.find(c => `候选 ${c.id}` === selected.label);
        if (!selectedCandidate || !selectedCandidate.code) {
            vscode.window.showWarningMessage('候选代码无效');
            return;
        }

        editor.edit(editBuilder => {
            const position = editor.selection.active;
            editBuilder.insert(position, selectedCandidate.code);
        });

        vscode.window.showInformationMessage('✅ 代码已插入！');
    });

    // -----------------------------------------------------------------------
    // 功能4：一键测试完整流程
    // -----------------------------------------------------------------------
    let testFullFlow = vscode.commands.registerCommand('my-smart-coding-plugin.testFullFlow', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个代码文件！');
            return;
        }
        if (!API_KEY) {
            vscode.window.showErrorMessage('请先在 .env 文件里配置 AI_API_KEY');
            return;
        }

        console.log('🧪 开始完整流程测试...');
        let contextData;
        try {
            contextData = await getEditorContext({ mode: "partial", beforeLines: 50, afterLines: 50 });
            console.log('✅ 第一步完成：上下文获取成功');
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
                const backend = getModelBackend();
                const aiConfig = {
                    apiKey: API_KEY,
                    candidateCount: 2,
                    model: backend === 'local' ? 'qwen-coder-l2' : "qwen-plus"
                };
                if (backend === 'local') {
                    aiConfig.endpoint = getLocalEndpoint();
                }
                aiResult = await generateRecommendedCode(contextData, aiConfig);
                console.log('✅ 第二步完成：AI生成成功');
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

    // -----------------------------------------------------------------------
    // 功能5（D7）：教学引导模式 — 张瑞泽实现
    // 根据 hitsmartcoder.completionMode 配置，输出不同深度的教学引导
    // -----------------------------------------------------------------------
    let teachingGuide = vscode.commands.registerCommand('my-smart-coding-plugin.teachingGuide', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('请先打开一个代码文件！');
            return;
        }

        const mode = getCompletionMode();
        if (mode === 'standard') {
            // standard 模式下提示用户切换模式
            const switchMode = await vscode.window.showInformationMessage(
                '当前是标准补全模式，要切换到教学引导模式吗？',
                '切换为 hint（提示）',
                '切换为 explain（讲解）',
                '不用'
            );
            if (switchMode === '切换为 hint（提示）') {
                await vscode.workspace.getConfiguration().update('hitsmartcoder.completionMode', 'hint', true);
                vscode.window.showInformationMessage('已切换为 hint 模式，请再次运行教学引导命令');
            } else if (switchMode === '切换为 explain（讲解）') {
                await vscode.workspace.getConfiguration().update('hitsmartcoder.completionMode', 'explain', true);
                vscode.window.showInformationMessage('已切换为 explain 模式，请再次运行教学引导命令');
            }
            return;
        }

        if (!API_KEY) {
            vscode.window.showErrorMessage('请先在 .env 文件中配置 AI_API_KEY');
            return;
        }

        let contextData;
        try {
            contextData = await getEditorContext({
                mode: "partial",
                beforeLines: 50,
                afterLines: 50
            });
        } catch (err) {
            vscode.window.showErrorMessage('上下文获取失败：' + err.message);
            return;
        }
        if (contextData.error && contextData.error.trim() !== "") {
            vscode.window.showErrorMessage('上下文错误：' + contextData.error);
            return;
        }

        const task = await vscode.window.showInputBox({
            prompt: '你想实现什么功能？AI 将引导你（不是直接给答案）',
            placeHolder: '例如：实现二分查找、写一个快速排序',
            value: '补全当前代码'
        });
        if (!task) { return; }
        contextData.task = task;

        let guideResult;
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `AI 正在生成 ${mode} 模式教学引导...`
        }, async () => {
            try {
                guideResult = await generateTeachingGuide(
                    {
                        code: contextData.partialCode ? contextData.partialCode.beforeCursor : '',
                        task: contextData.task,
                        language: contextData.language
                    },
                    {
                        apiKey: API_KEY,
                        guideLevel: mode,
                        includeExample: false,
                        temperature: 0.3
                    }
                );
            } catch (err) {
                vscode.window.showErrorMessage('教学引导生成失败：' + err.message);
            }
        });

        if (!guideResult) { return; }
        if (guideResult.error && guideResult.error.trim() !== "") {
            vscode.window.showErrorMessage(guideResult.error);
            return;
        }

        // 组装展示内容
        let displayText = `📚 ${mode === 'hint' ? '提示引导' : mode === 'scaffold' ? '框架引导' : '讲解引导'}\n`;
        displayText += `─${'─'.repeat(30)}\n`;
        displayText += guideResult.guide || '（无引导内容）';
        if (guideResult.concepts && guideResult.concepts.length > 0) {
            displayText += `\n\n📌 涉及知识点：\n${guideResult.concepts.map(c => `• ${c}`).join('\n')}`;
        }
        if (guideResult.questions && guideResult.questions.length > 0) {
            displayText += `\n\n🤔 思考题：\n${guideResult.questions.map(q => `• ${q}`).join('\n')}`;
        }

        // 用 Markdown 弹窗展示
        const panel = vscode.window.createWebviewPanel(
            'teachingGuide',
            `教学引导 - ${mode}`,
            vscode.ViewColumn.Beside,
            {}
        );
        panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 20px; line-height: 1.6; }
h2 { color: #004DA0; border-bottom: 2px solid #004DA0; padding-bottom: 8px; }
.concept { background: #E8F0FE; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
.question { background: #FFF3E0; padding: 10px 15px; border-radius: 8px; margin: 10px 0; }
.guide { font-size: 15px; }
.tag { display: inline-block; background: #004DA0; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; }
</style></head><body>
<h2>📚 教学引导 <span class="tag">${mode}</span></h2>
<div class="guide">${guideResult.guide.replace(/\n/g, '<br>')}</div>
${guideResult.concepts.length ? `<div class="concept"><b>📌 知识点</b><br>${guideResult.concepts.map(c => '• ' + c).join('<br>')}</div>` : ''}
${guideResult.questions.length ? `<div class="question"><b>🤔 思考题</b><br>${guideResult.questions.map(q => '• ' + q).join('<br>')}</div>` : ''}
</body></html>`;
    });

    // ================================================================
    // 功能5（D6）：自动触发补全（Inline Completion）— 郭鑫龙实现
    // ================================================================
    const inlineProvider = createInlineCompletionProvider();

    // ================================================================
    // 功能6（D10）：代码高亮 Diff 预览 — 郭鑫龙实现
    // ================================================================
    const highlightCmd = createHighlightCommand();

    // ---------- 注册所有功能 ----------
    context.subscriptions.push(
        insertLog,
        explainCode,
        smartComplete,
        testFullFlow,
        teachingGuide,
        inlineProvider,
        highlightCmd
    );

    vscode.window.showInformationMessage('✅ HIT 智能编码助手已加载（含自动补全、Diff预览、教学引导）');
    console.log('✅ 所有功能注册完成');
}

function deactivate() {}

module.exports = { activate, deactivate };
