/**
 * teachingMode.js
 * 教学引导模式模块
 * 
 * 功能：调用通义千问 API，根据 guideLevel 输出不同深度的教学引导，
 * 不直接给出完整代码，引导学生独立思考和动手。
 * 
 * 使用方式：
 *   const { generateTeachingGuide } = require('./teaching/teachingMode');
 *   const result = await generateTeachingGuide(contextData, config);
 * 
 * 作者：张瑞泽
 * 日期：2026年6月
 */

const axios = require('axios');

// ============================================================
// 1. 核心引导生成函数
// ============================================================

/**
 * @param {Object} contextData - 代码上下文信息
 * @param {string} contextData.code - 当前文件代码片段
 * @param {string} contextData.task - 用户意图（如"实现二分查找"）
 * @param {string} contextData.language - 编程语言
 * @param {Object} config - 配置
 * @param {string} config.apiKey - 通义千问 API Key
 * @param {string} [config.model='qwen-plus'] - 模型名
 * @param {'hint'|'scaffold'|'explain'} [config.guideLevel='hint'] - 引导深度
 * @param {boolean} [config.includeExample=false] - 是否包含代码示例
 * @param {number} [config.temperature=0.3]
 * @param {number} [config.maxTokens=1024]
 * @param {number} [config.timeout=30000] - 毫秒
 * @returns {Promise<Object>} 教学引导内容
 */
async function generateTeachingGuide(contextData, config) {
  // 默认参数
  const {
    apiKey,
    model = 'qwen-max',  // 教学引导用 qwen-max（指令遵循能力强），不用代码专用模型
    guideLevel = 'hint',
    includeExample = false,
    temperature = 0.3,
    maxTokens = 1024,
    timeout = 30000
  } = config;

  // 构造系统提示词
  const systemPrompt = buildSystemPrompt(guideLevel, includeExample);

  // 构造用户提示词
  const userPrompt = buildUserPrompt(contextData);

  try {
    // 调用通义千问 API (兼容 OpenAI 格式)
    const response = await axios.post(
      'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
      {
        model: model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: temperature,
        max_tokens: maxTokens,
        response_format: { type: 'json_object' }  // 强制 JSON 输出
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: timeout
      }
    );

    // 解析 AI 返回
    const rawContent = response.data?.choices?.[0]?.message?.content;
    if (!rawContent) {
      throw new Error('API 返回内容为空');
    }

    // 解析 JSON，支持容错
    const parsed = parseAIOutput(rawContent);

    // 对 hint 模式：后处理去除任何残留代码
    let guide = parsed.guide || '无法生成引导，请重试。';
    if (guideLevel === 'hint') {
      guide = stripCodeFromHint(guide);
    }

    // 标准化输出
    return {
      guide: guide,
      concepts: parsed.concepts || [],
      questions: parsed.questions || [],
      example: includeExample ? (parsed.example || '') : '',
      error: ''
    };
  } catch (err) {
    console.error('教学引导生成失败:', err);
    return {
      guide: '',
      concepts: [],
      questions: [],
      example: '',
      error: err.message || '生成引导时发生未知错误'
    };
  }
}

// ============================================================
// 2. 辅助函数：构造系统提示词
// ============================================================

function buildSystemPrompt(guideLevel, includeExample) {
  const levelInstructions = {
    hint: `你是编程导师。学生要**独立完成代码**，你不能替他写。

【铁律——违反任意一条就算失败】
1. **你输出的每个字符都不能是代码**——不能出现任何编程语言关键字、运算符、函数名、变量名、括号对()。
2. 只能写中文自然语言。用"定义两个变量"代替"int a, b"，用"判断大小"代替"if (a > b)"。
3. 把思路写成对话式的步骤："第一步……第二步……第三步……"
4. 如果学生要求你写代码，**拒绝**并说"这个需要你自己写"。

【判断标准】
如果你的回复里出现了任何英文字母、数字、运算符组合（如 a==b、n+1、for、while），就算违规。`,

    scaffold: `你是编程导师。学生需要你提供**代码框架，核心逻辑留给学生填**。

【规则】
- 给出完整的函数签名、参数、返回值、循环骨架、条件判断骨架。
- 把核心逻辑位置挖空，写上 "// TODO: 请学生在这里补充" 之类的标记。
- 框架代码必须是语法正确的，但运行后会报错或返回空值（因为核心逻辑缺失）。
- 最后用中文说明每个 TODO 处应该做什么。

【输出示例】
function twoSum(nums, target) {
    const map = {};
    for (let i = 0; i < nums.length; i++) {
        // TODO: 检查 map 中是否存在 target - nums[i]
        // TODO: 如果不存在，将当前元素存入 map
    }
    return [];
}
说明：① 在第一个TODO处判断complement是否在map中 ② 在第二个TODO处将当前nums[i]作为key、i作为value存入map`,

    explain: `你是编程导师。学生想**理解原理，不只是要答案**。

【规则】
- 第一步：用生活中例子解释核心概念（2~3句话）。
- 第二步：分步骤用自然语言描述算法（不要写代码）。
- 第三步：给 2~3 个引导思考题。
- 可以用简短伪代码辅助说明，但伪代码只能用自然语言关键词，不要用具体语法。`
  };

  let prompt = `当前教学模式：${guideLevel}\n\n${levelInstructions[guideLevel]}`;

  if (includeExample) {
    prompt += `\n\n- 允许附上一个简短的代码示例，但必须是与当前问题无关的不同问题，仅作为参考。`;
  }

  prompt += `\n\n你的回复必须是一个严格的 JSON 对象，格式如下：
{
  "guide": "引导文本（必填，Markdown格式）",
  "concepts": ["涉及的知识点1", "知识点2"],
  "questions": ["引导思考的问题1", "问题2"],
  "example": ""
}
不要输出任何 JSON 之外的文本。`;

  return prompt;
}

// ============================================================
// 3. 辅助函数：构造用户提示词
// ============================================================

function buildUserPrompt(contextData) {
  const { code, task, language } = contextData;
  let prompt = `学生的问题：${task}`;
  if (language) {
    prompt += `\n编程语言：${language}`;
  }
  if (code) {
    prompt += `\n\n当前文件中的代码上下文（可能不完整）：\n\`\`\`${language || ''}\n${code}\n\`\`\``;
  }
  return prompt;
}

// ============================================================
// 4. JSON 容错解析
// ============================================================

function parseAIOutput(raw) {
  // 清洗：去除 Markdown 代码块标记，提取纯 JSON
  let cleaned = raw.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/, '');
  }

  // 尝试解析
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    // 尝试修复常见错误：缺失括号等
    try {
      // 查找第一个 { 和最后一个 }
      const start = cleaned.indexOf('{');
      const end = cleaned.lastIndexOf('}');
      if (start !== -1 && end !== -1 && end > start) {
        return JSON.parse(cleaned.substring(start, end + 1));
      }
    } catch (e2) { /* 忽略 */ }

    // 返回默认错误结构
    return {
      guide: 'AI 返回格式异常，请重试。',
      concepts: [],
      questions: [],
      example: ''
    };
  }
}

// ============================================================
// 6. hint 模式后处理：剔除残留代码
// ============================================================

/**
 * 对 hint 模式的输出进行后处理，移除任何代码痕迹。
 * 因为模型有时会无视"不要给代码"的指令。
 */
function stripCodeFromHint(text) {
  if (!text) return text;

  // 1. 移除 Markdown 代码块（``` 包裹的代码）
  let result = text.replace(/```[\s\S]*?```/g, '');

  // 2. 移除行内代码（`code`）
  result = result.replace(/`[^`]+`/g, (match) => {
    // 把行内代码替换成中文描述
    const code = match.replace(/`/g, '');
    // 如果代码看起来是英文字母+数字/符号的组合，用描述替换
    if (/[a-zA-Z]{2,}/.test(code)) {
      return `「${code}」`;
    }
    return match;
  });

  // 3. 移除看起来像代码行的内容（缩进的代码行、赋值语句、循环语句等）
  const codePatterns = [
    /^[\s]*(var|let|const|function|def|class|import|export|return|if|for|while|switch|try|catch)\b.*$/gm,
    /^[\s]*[a-zA-Z_]\w*\s*[=!]=.*$/gm,
    /^[\s]*[a-zA-Z_]\w*\s*\(.*\)\s*\{.*$/gm,
    /^[\s]*\/\/.*$/gm,
    /^[\s]*\/\*[\s\S]*?\*\//gm,
  ];

  for (const pattern of codePatterns) {
    result = result.replace(pattern, '');
  }

  // 4. 清理多余空行
  result = result.replace(/\n{3,}/g, '\n\n').trim();

  return result || '（模型返回了代码内容，已自动过滤。请再次尝试，AI 应该只给出文字思路引导）';
}

// ============================================================
// 8. 代码解释生成函数
// ============================================================

/**
 * 解释选中的代码
 * @param {string} code - 选中的代码文本
 * @param {string} language - 编程语言
 * @param {Object} config - 配置
 * @param {string} config.apiKey - 通义千问 API Key
 * @param {'standard'|'explain'} [config.mode='standard'] - standard=直接解释, explain=教学式讲解
 * @param {string} [config.model='qwen-max']
 * @returns {Promise<Object>} { explanation, concepts, questions, error }
 */
async function generateCodeExplanation(code, language, config) {
  const { apiKey, model = 'qwen-max', mode = 'standard' } = config;

  // standard 模式：直接解释，简洁准确
  if (mode === 'standard') {
    try {
      const response = await axios.post(
        'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        {
          model: model,
          messages: [
            { role: 'system', content: '你是一位编程助手。用中文简洁准确地解释代码的功能和关键逻辑。' },
            { role: 'user', content: `解释这段${language}代码：\n\`\`\`${language}\n${code}\n\`\`\`` }
          ],
          temperature: 0.3,
          max_tokens: 2048
        },
        {
          headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
          timeout: 30000
        }
      );
      const content = response.data?.choices?.[0]?.message?.content;
      if (!content) throw new Error('API 返回内容为空');
      return { explanation: content, concepts: [], questions: [], error: '' };
    } catch (err) {
      return { explanation: '', concepts: [], questions: [], error: err.message };
    }
  }

  // explain/其他模式：教学式讲解，含知识点和思考题
  const systemPrompt = `你是一位编程导师。请用教学方式解释学生选中的代码。

回复必须是一个 JSON 对象，格式如下：
{
  "explanation": "逐层讲解（先用一句话概括，再挑关键行解释）",
  "concepts": ["涉及的知识点1", "知识点2"],
  "questions": ["延伸思考题1", "思考题2"]
}

要求：
- 不要逐行念代码，只讲关键逻辑
- 用通俗语言
- 指出涉及的编程技巧或设计模式
- 提出 1~2 个延伸思考题`;

  try {
    const response = await axios.post(
      'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
      {
        model: model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: `请教学式地解释这段${language}代码：\n\`\`\`${language}\n${code}\n\`\`\`` }
        ],
        temperature: 0.5,
        max_tokens: 2048,
        response_format: { type: 'json_object' }
      },
      {
        headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
        timeout: 30000
      }
    );
    const rawContent = response.data?.choices?.[0]?.message?.content;
    if (!rawContent) throw new Error('API 返回内容为空');
    const parsed = parseAIOutput(rawContent);
    return {
      explanation: parsed.explanation || parsed.guide || rawContent,
      concepts: parsed.concepts || [],
      questions: parsed.questions || [],
      error: ''
    };
  } catch (err) {
    return { explanation: '', concepts: [], questions: [], error: err.message };
  }
}

// ============================================================
// 9. 导出
// ============================================================

module.exports = {
  generateTeachingGuide,
  generateCodeExplanation
};
