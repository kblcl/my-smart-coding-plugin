// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    // 功能1：选中变量，插入 console.log
    let insertLog = vscode.commands.registerCommand('my-ai-coder.insertLog', function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) { return; }

        const document = editor.document;
        const selection = editor.selection;
        const selectedText = document.getText(selection); // 获取选中的文本

        if (!selectedText) {
            vscode.window.showWarningMessage('请先选中一个变量名！');
            return;
        }

        // 构造 log 语句
        const logStatement = `console.log('${selectedText}:', ${selectedText});`;

        // 插入到编辑器
        editor.edit(editBuilder => {
            editBuilder.replace(selection, logStatement);
        });
    });

    context.subscriptions.push(insertLog);
}

module.exports = { activate };