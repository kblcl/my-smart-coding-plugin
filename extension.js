const vscode = require('vscode');
const axios = require('axios');
const path = require('path');

function activate(context) {
    // 1. 加载环境变量（确保在 activate 内部）
    require('dotenv').config({ path: path.join(context.extensionPath, '.env') });

    // 2. 读取配置
    const API_KEY = process.env.AI_API_KEY;
    const BASE_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation';
    const MODEL_NAME = 'qwen-turbo';

    // 调试：确认配置加载
    console.log('插件已激活');
    console.log('API Key 状态:', API_KEY ? '已加载' : '未加载');

    // 功能1: 插入 console.log
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

    // 功能2: AI 解释代码（核心）
    let explainCode = vscode.commands.registerCommand('my-smart-coding-plugin.explainCode', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return; }

        const selectedCode = editor.document.getText(editor.selection);
        if (!selectedCode) {
            vscode.window.showWarningMessage('请先选中一段代码！');
            return;
        }

        // 检查 API Key
        if (!API_KEY) {
            vscode.window.showErrorMessage('请先在 .env 文件中配置 AI_API_KEY');
            return;
        }

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "AI 正在分析代码..."
        }, async () => {
            try {
                const prompt = `请用通俗易懂的语言解释以下代码的功能：\n${selectedCode}`;
                
                console.log('正在调用通义千问 API...');
                
                const response = await axios.post(
                    BASE_URL,
                    {
                        model: MODEL_NAME,
                        input: {
                            messages: [{ role: 'user', content: prompt }]
                        },
                        parameters: {
                            result_format: 'message'
                        }
                    },
                    {
                        headers: {
                            'Authorization': `Bearer ${API_KEY}`,
                            'Content-Type': 'application/json'
                        }
                    }
                );

                // 解析通义千问的回复
                const aiAnswer = response.data.output.choices[0].message.content;
                vscode.window.showInformationMessage(aiAnswer, { modal: true });

            } catch (error) {
                console.error('AI 调用失败:', error.response ? error.response.data : error.message);
                const errorMsg = error.response ? 
                    `AI 调用失败: ${error.response.status} - ${JSON.stringify(error.response.data)}` :
                    `AI 调用失败: ${error.message}`;
                vscode.window.showErrorMessage(errorMsg);
            }
        });
    });

    context.subscriptions.push(insertLog);
    context.subscriptions.push(explainCode);
}

function deactivate() {}

module.exports = { activate, deactivate };