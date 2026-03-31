const vscode = require('vscode');

/**
 * 获取编辑器里的代码上下文
 * @param {Object} config 配置参数(格式固定,见协议文档)
 * @returns {Promise<Object>} 固定格式的上下文数据
 */
async function getEditorContext(config) {
    // 全局错误捕获，杜绝代码崩溃，符合协议强制错误处理要求
    try {
        // 1. 入参合法性校验（严格遵循协议入参规范）
        if (!config || typeof config !== 'object' || Array.isArray(config)) {
            return {
                language: '',
                cursorPosition: { line: 0, column: 0 },
                error: '入参config必须是合法的JS对象'
            };
        }

        const { mode } = config;
        if (!mode || !['partial', 'full'].includes(mode)) {
            return {
                language: '',
                cursorPosition: { line: 0, column: 0 },
                error: 'mode字段必须为"partial"或"full"'
            };
        }

        let beforeLines = 50;
        let afterLines = 50;
        if (mode === 'partial') {
            if (config.beforeLines === undefined || config.afterLines === undefined) {
                return {
                    language: '',
                    cursorPosition: { line: 0, column: 0 },
                    error: 'mode=partial时，beforeLines和afterLines为必填字段'
                };
            }
            if (!Number.isInteger(config.beforeLines) || config.beforeLines <= 0 || !Number.isInteger(config.afterLines) || config.afterLines <= 0) {
                return {
                    language: '',
                    cursorPosition: { line: 0, column: 0 },
                    error: 'mode=partial时，beforeLines和afterLines必须为正整数'
                };
            }
            beforeLines = config.beforeLines;
            afterLines = config.afterLines;
        }

        // 2. 获取编辑器核心实例
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return {
                language: '',
                cursorPosition: { line: 0, column: 0 },
                error: '用户未打开任何编辑文件，请先打开一个代码文件'
            };
        }
        const document = editor.document;
        const cursor = editor.selection.active;

        // 3. 基础返回字段（严格遵循协议字段名与格式）
        const baseResult = {
            language: document.languageId,
            cursorPosition: {
                line: cursor.line,
                column: cursor.character
            },
            error: ''
        };

        // 4. 处理full模式（严格遵循协议要求，返回完整文件代码）
        if (mode === 'full') {
            return {
                ...baseResult,
                fullCode: document.getText(),
                error: ''
            };
        }

        // 5. partial模式核心逻辑：优先识别完整函数/类，降级为按行截取
        let targetBlockRange = null;

        try {
            // 调用VS Code官方符号API，兼容所有VS Code支持的编程语言
            const symbols = await vscode.commands.executeCommand('vscode.executeDocumentSymbolProvider', document.uri);
            
            if (symbols && Array.isArray(symbols) && symbols.length > 0) {
                // 递归查找包含光标、层级最深的目标符号（函数>方法>构造器>类）
                const findTargetSymbol = (symbolList, parentMatch = null) => {
                    for (const symbol of symbolList) {
                        if (symbol.range.contains(cursor)) {
                            // 匹配协议要求的目标类型
                            const isValidType = [
                                vscode.SymbolKind.Function,
                                vscode.SymbolKind.Method,
                                vscode.SymbolKind.Constructor,
                                vscode.SymbolKind.Class
                            ].includes(symbol.kind);
                            
                            let currentMatch = isValidType ? symbol : parentMatch;

                            // 递归遍历子符号，优先取最深层级的匹配
                            if (symbol.children && symbol.children.length > 0) {
                                const childMatch = findTargetSymbol(symbol.children, currentMatch);
                                if (childMatch) currentMatch = childMatch;
                            }

                            return currentMatch;
                        }
                    }
                    return null;
                };

                const matchedSymbol = findTargetSymbol(symbols);
                if (matchedSymbol) targetBlockRange = matchedSymbol.range;
            }
        } catch (symbolError) {
            // 符号识别失败时降级处理，不中断主流程
            console.warn('代码块识别失败，已降级为按行截取：', symbolError);
        }

        // 6. 确定最终截取范围
        let finalRange;
        if (targetBlockRange) {
            // 识别到函数/类，取完整代码块
            finalRange = targetBlockRange;
        } else {
            // 未识别到，按协议配置的前后行截取，处理边界越界
            const maxLine = document.lineCount - 1;
            const startLine = Math.max(0, cursor.line - beforeLines);
            const endLine = Math.min(maxLine, cursor.line + afterLines);
            finalRange = new vscode.Range(
                startLine, 0,
                endLine, document.lineAt(endLine).text.length
            );
        }

        // 7. 按光标位置拆分代码，严格遵循协议partialCode格式
        const fullBlockText = document.getText(finalRange);
        const cursorOffset = document.offsetAt(cursor);
        const rangeStartOffset = document.offsetAt(finalRange.start);
        const splitIndex = cursorOffset - rangeStartOffset;

        // 8. 组装最终返回值，100%匹配协议字段规范
        return {
            ...baseResult,
            partialCode: {
                beforeCursor: fullBlockText.slice(0, splitIndex),
                afterCursor: fullBlockText.slice(splitIndex)
            },
            error: ''
        };

    } catch (fatalError) {
        // 兜底错误处理，完全符合协议强制错误处理要求
        console.error('getEditorContext执行异常：', fatalError);
        return {
            language: '',
            cursorPosition: { line: 0, column: 0 },
            error: `代码上下文获取失败: ${fatalError.message || '未知错误'}`
        };
    }
}

// 严格遵循协议要求的导出方式，禁止修改
module.exports = { getEditorContext };