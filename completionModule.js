// completionModule.js - 改用阿里云通义千问
const axios = require('axios');

// ---------------------------------------------------------------------------
// 系统提示词构建（动态候选数）
// ---------------------------------------------------------------------------

function buildSystemPrompt(candidateCount) {
  return `You are an expert code assistant. Your task is to generate code suggestions.

CRITICAL INSTRUCTION:
- Your entire response must be a single, complete JSON object with exactly the structure below.
- Do NOT include any text before or after the JSON.
- Do NOT wrap the JSON in markdown code blocks.
- Ensure all brackets are properly closed.

RESPONSE FORMAT:
{
  "candidates": [
    { "id": 1, "code": "<code here>", "description": "<10-30 words in Chinese>" }
  ]
}

Rules:
- "code" must be clean, runnable code — no markdown fences, no language labels, no extra comments.
- "description" must be written in Chinese (简体中文) and be 10–30 words long, explaining what the code does.
- Always return exactly ${candidateCount} candidates, even if they are variations of the same idea.
- Never include anything outside the JSON object.
- **When the function name indicates a well-known algorithm (e.g., bubble, quickSort, binarySearch, fibonacci, factorial, reverse, mergeSort), you MUST generate the standard implementation of that algorithm in the given language, using the provided parameter names.**
- **If the code is in partial mode (cursor inside a function body), you must complete the function body with the appropriate logic, not just variable declarations.**
- **Do not output placeholder or generic code like "// ..." – provide actual runnable code.**`;
}

// ---------------------------------------------------------------------------
// 用户提示词构建（适配协议 contextData 格式）
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 修复版：用户提示词构建，确保把完整代码上下文传给AI
// ---------------------------------------------------------------------------
function buildUserPrompt(contextData, candidateCount) {
  const { language, task, mode, partialCode, fullCode, cursorPosition } = contextData;

  // 提取函数名和参数（partial 和 full 模式都尝试，支持多语言语法）
  let functionName = "";
  let functionParams = "";
  const codeToSearch = (partialCode && partialCode.beforeCursor)
    ? partialCode.beforeCursor
    : (fullCode || "");

  const funcPatterns = [
    /function\s+(\w+)\s*\(([^)]*)\)/,                                      // JS: function bubble(arr)
    /def\s+(\w+)\s*\(([^)]*)\)/,                                           // Python: def bubble(arr)
    /(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>/,     // arrow: const bubble = (arr) =>
    /(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{/,                              // method shorthand: bubble(arr) {
  ];
  for (const pattern of funcPatterns) {
    const m = codeToSearch.match(pattern);
    if (m) {
      functionName = m[1];
      functionParams = (m[2] || "").trim();
      break;
    }
  }

  // 已知算法映射
  const algorithmMap = {
    bubble: "冒泡排序",
    quickSort: "快速排序",
    binarySearch: "二分查找",
    fibonacci: "斐波那契数列",
    factorial: "阶乘",
    reverse: "反转数组/字符串"
  };
  let algoHint = "";
  const algoName = functionName && algorithmMap[functionName];
  if (algoName) {
    algoHint = `\n⚠️ **重要**：函数名 "${functionName}" 表示要实现 ${algoName} 算法。\n请生成完整的 ${algoName} 实现，包含循环、条件判断等必要逻辑，而不是简单的返回语句。\n`;
  }

  // 当检测到已知算法时，用算法描述覆盖用户输入的通用 task，避免 AI 忽略算法提示
  const effectiveTask = algoName
    ? `实现完整的 ${algoName} 算法`
    : (task || "补全光标位置的代码，保持和前后风格一致");

  let prompt = `【核心要求】
你是专业的代码补全助手，必须严格根据我提供的「光标前后的代码」，在光标位置生成符合语法、风格一致的可运行代码。

【基础信息】
- 编程语言：${language}
- 用户需求：${effectiveTask}
- 光标位置：第 ${cursorPosition ? cursorPosition.line + 1 : "未知"} 行，第 ${cursorPosition ? cursorPosition.column + 1 : "未知"} 列
`;

  if (functionName) {
    prompt += `- 当前函数名：${functionName}\n`;
    if (functionParams) prompt += `- 函数参数：${functionParams}\n`;
  }
  prompt += algoHint;

  prompt += "\n";

  if (partialCode) {
    const before = partialCode.beforeCursor || "";
    const after = partialCode.afterCursor || "";
    if (before || after) {
      prompt += `==================== 光标前的代码 ====================\n${before}\n\n`;
      prompt += `==================== 光标后的代码 ====================\n${after}\n\n`;
      prompt += `【强制规则】
1. 你的代码必须插入在「光标前的代码」和「光标后的代码」之间；
2. 必须和前后代码的语法、缩进、变量名、风格完全一致；
3. 绝对不能重复前后已经存在的代码；
4. 只返回需要插入的代码部分，不要把前后代码再包一遍。
`;
      if (functionName && algorithmMap[functionName]) {
        prompt += `5. 对于算法函数，必须生成完整的循环和逻辑，不能只写一行返回语句。\n`;
      }
    }
  } else if (fullCode && fullCode.trim().length > 0) {
    prompt += `==================== 当前完整代码 ====================\n${fullCode}\n\n`;
  }

  prompt += `请严格按照系统提示词的要求，生成 ${candidateCount} 个代码候选，只返回符合格式的JSON。`;

  // 新增强化指令
  prompt += `\n\nIMPORTANT: Your output must be ONLY the complete JSON object, with no extra text. Ensure all brackets are closed.`;

  console.log('📝 最终给AI的用户提示词完整内容：\n', prompt);
  return prompt;
}

// ---------------------------------------------------------------------------
// 代码清理（去除 markdown 标记）
// ---------------------------------------------------------------------------

function cleanCode(raw) {
  return raw
    .replace(/^```[\w]*\n?/gm, "")
    .replace(/^```$/gm, "")
    .trim();
}

// ---------------------------------------------------------------------------
// description 长度校验（10-30字）
// ---------------------------------------------------------------------------

function normalizeDescription(desc) {
  if (!desc || desc.trim().length === 0) return "该代码片段实现了指定功能，可直接插入编辑器使用。";
  const trimmed = desc.trim();
  if (trimmed.length > 30) return trimmed.slice(0, 30);
  if (trimmed.length < 10) return trimmed + "，可直接插入编辑器使用。".slice(0, 10 - trimmed.length + 10);
  return trimmed;
}

// ---------------------------------------------------------------------------
// 响应解析（严格按协议格式输出）
// ---------------------------------------------------------------------------

function parseResponse(rawText, candidateCount) {
  // 1. 去除首尾空白字符
  let cleaned = rawText.trim();
  
  // 2. 去除可能的 markdown 代码块标记
  cleaned = cleaned.replace(/^```json\s*\n?/i, '').replace(/\n?```\s*$/i, '');
  cleaned = cleaned.trim();

  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (e) {
    // 尝试用正则提取最外层 JSON 对象（避免 code 字段内的括号干扰计数）
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { candidates: [], error: "AI响应解析失败，返回内容无法识别" };
    }
    try {
      parsed = JSON.parse(jsonMatch[0]);
    } catch {
      return { candidates: [], error: "AI响应解析失败，JSON格式错误" };
    }
  }

  if (!Array.isArray(parsed.candidates)) {
    return { candidates: [], error: "AI响应解析失败，缺少candidates字段" };
  }

  let candidates = parsed.candidates.map((c, i) => ({
    id: i + 1,
    code: cleanCode(String(c.code ?? "")),
    description: normalizeDescription(String(c.description ?? "")),
  }));

  if (candidates.length < candidateCount) {
    for (let i = candidates.length; i < candidateCount; i++) {
      candidates.push({ id: i + 1, code: "", description: "无推荐代码" });
    }
  } else if (candidates.length > candidateCount) {
    candidates = candidates.slice(0, candidateCount);
  }

  return { candidates, error: "" };
}

// ---------------------------------------------------------------------------
// 错误映射（中文场景化提示 - 通义千问版本）
// ---------------------------------------------------------------------------

function mapApiError(error) {
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;
    if (status === 401) return "AI接口调用失败，API Key无效或缺失";
    if (status === 429) return "AI接口调用失败，请求频率超限，请稍后重试";
    if (status === 400) return `AI接口调用失败，请求参数错误: ${JSON.stringify(data)}`;
    return `AI接口调用失败，HTTP ${status}: ${JSON.stringify(data)}`;
  }
  if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
    return "AI接口调用失败，网络异常，请检查网络连接";
  }
  if (error.code === 'ETIMEDOUT') {
    return "AI接口调用失败，请求超时，请稍后重试";
  }
  return `AI接口调用失败，未知错误: ${error.message}`;
}

// ---------------------------------------------------------------------------
// 核心对外接口（已改为通义千问）
// ---------------------------------------------------------------------------

/**
 * 代码补全与生成模块的唯一对外接口。
 * @param {object} contextData - 上下文模块返回的原格式对象
 * @param {object} aiConfig - AI配置对象
 * @param {string} aiConfig.apiKey - 阿里云通义千问 API Key
 * @param {number} aiConfig.candidateCount - 候选代码数量（正整数）
 * @param {string} [aiConfig.model] - 可选，模型名称，默认 qwen-turbo
 * @returns {Promise<{ candidates: Array<{id: number, code: string, description: string}>, error: string }>}
 */
async function generateRecommendedCode(contextData, aiConfig) {
  // 参数校验
  if (
    !aiConfig ||
    typeof aiConfig.apiKey !== "string" ||
    aiConfig.apiKey.trim() === "" ||
    !Number.isInteger(aiConfig.candidateCount) ||
    aiConfig.candidateCount <= 0
  ) {
    return { candidates: [], error: "AI配置错误，缺少有效密钥或候选数" };
  }

  if (!contextData || !contextData.language) {
    return { candidates: [], error: "上下文数据无效，缺少language字段" };
  }

  // 上下文模块报错（如文件打不开），直接透传，不调用 AI
  if (contextData.error && contextData.error.trim() !== "") {
    return { candidates: [], error: `上下文错误：${contextData.error}` };
  }

  const { candidateCount, apiKey, model = "qwen-turbo" } = aiConfig;
  const baseURL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation";

  // 构建完整的用户消息（包含系统指令+用户提示）
  const systemPrompt = buildSystemPrompt(candidateCount);
  const userPrompt = buildUserPrompt(contextData, candidateCount);
  // 将系统提示词放到 messages 的开头
// 将系统提示词放到 messages 的开头
  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userPrompt }
  ];

  // ======================
  // 新增：打印完整请求，看 AI 到底收到了什么
  // ======================
  console.log('🔍 最终发给 AI 的完整消息：');
  console.log(JSON.stringify(messages, null, 2));

  try {
    const response = await axios.post(
      baseURL,
      {
        model: model,
        input: { messages: messages },
        parameters: {
          result_format: "message",
          max_tokens: 4096,
          temperature: 0.6   
        }
      },
      {
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        timeout: 30000  // 30秒超时
      }
    );

    // 通义千问返回格式解析
    const aiMessage = response.data.output?.choices?.[0]?.message;
    if (!aiMessage || !aiMessage.content) {
      return { candidates: [], error: "AI返回空响应，未获取到任何内容" };
    }

    const rawText = aiMessage.content;
    console.log('AI原始返回内容：\n',rawText)
    return parseResponse(rawText, candidateCount);
  

  } catch (error) {
    return { candidates: [], error: mapApiError(error) };
  }
}

// ---------------------------------------------------------------------------
// 测试示例（本地测试用）
// ---------------------------------------------------------------------------
// async function example() {
//   // 测试场景：生成 Python 冒泡排序代码
//   const contextDataFull = {
//     language: "python", // 换语言测试
//     task: "生成一个完整的冒泡排序函数，包含测试用例",
//     mode: "full", // 完整代码模式
//     fullCode: "", // 无现有代码
//     cursorPosition: { line: 0, column: 0 },
//     error: ""
//   };

//   const aiConfig = {
//     candidateCount: 3, // 生成 3 个候选
//     apiKey: process.env.AI_API_KEY,
//     model: "qwen-turbo"
//   };

//   console.log("--- 测试完整代码生成（Python） ---");
//   const result = await generateRecommendedCode(contextDataFull, aiConfig);
//   console.log(JSON.stringify(result, null, 2));
// }

// 本地测试取消注释运行：node completionModule.js
// 需要先设置环境变量 export DASHSCOPE_API_KEY="your-key"
//example();

module.exports = { generateRecommendedCode };