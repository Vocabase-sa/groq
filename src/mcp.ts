import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import {
  createChatCompletion,
  createEmbedding,
  getModel,
  groqConfig,
  listModels,
  normalizeContent,
  toChatMessage,
  type OllamaMessage
} from "./groq.js";

const server = new McpServer({
  name: "groq-mcp",
  version: "0.1.0"
});

server.registerTool(
  "list_models",
  {
    title: "List models",
    description: "Lists models available through the configured Groq provider.",
    inputSchema: {}
  },
  async () => {
    const models = await listModels();
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            provider: "groq",
            default_model: groqConfig.defaultModel,
            models: models.map((model) => model.id)
          }, null, 2)
        }
      ]
    };
  }
);

server.registerTool(
  "generate_text",
  {
    title: "Generate text",
    description: "Runs a single-prompt chat completion against the Groq API.",
    inputSchema: {
      model: z.string().optional(),
      prompt: z.string(),
      system: z.string().optional(),
      temperature: z.number().min(0).max(2).optional(),
      max_completion_tokens: z.number().int().positive().optional(),
      json_mode: z.boolean().optional()
    }
  },
  async ({ model, prompt, system, temperature, max_completion_tokens, json_mode }) => {
    const messages: ChatCompletionMessageParam[] = [];

    if (system) {
      messages.push({ role: "system", content: system });
    }

    messages.push({ role: "user", content: prompt });

    const completion = await createChatCompletion({
      model,
      messages,
      temperature,
      maxCompletionTokens: max_completion_tokens,
      jsonMode: json_mode
    });

    const text = completion.choices[0]?.message?.content ?? "";

    return {
      content: [
        {
          type: "text",
          text
        }
      ],
      structuredContent: {
        model: getModel(model),
        resolved_model: completion.model ?? getModel(model),
        output: text,
        finish_reason: completion.choices[0]?.finish_reason ?? "stop",
        usage: completion.usage ?? null
      }
    };
  }
);

server.registerTool(
  "chat_completion",
  {
    title: "Chat completion",
    description: "Runs a multi-turn chat completion against the Groq API.",
    inputSchema: {
      model: z.string().optional(),
      messages: z.array(
        z.object({
          role: z.enum(["system", "user", "assistant", "tool"]),
          content: z.string().optional(),
          images: z.array(z.string()).optional()
        })
      ).min(1),
      temperature: z.number().min(0).max(2).optional(),
      max_completion_tokens: z.number().int().positive().optional(),
      json_mode: z.boolean().optional()
    }
  },
  async ({ model, messages, temperature, max_completion_tokens, json_mode }) => {
    const completion = await createChatCompletion({
      model,
      messages: messages.map((message) => toChatMessage(message as OllamaMessage)),
      temperature,
      maxCompletionTokens: max_completion_tokens,
      jsonMode: json_mode
    });

    const text = completion.choices[0]?.message?.content ?? "";

    return {
      content: [
        {
          type: "text",
          text
        }
      ],
      structuredContent: {
        model: getModel(model),
        resolved_model: completion.model ?? getModel(model),
        assistant_message: {
          role: "assistant",
          content: text
        },
        normalized_messages: messages.map((message) => ({
          role: message.role,
          content: normalizeContent(message as OllamaMessage)
        })),
        finish_reason: completion.choices[0]?.finish_reason ?? "stop",
        usage: completion.usage ?? null
      }
    };
  }
);

server.registerTool(
  "embed_text",
  {
    title: "Embed text",
    description: "Generates vector embeddings for one or more input strings.",
    inputSchema: {
      model: z.string().optional(),
      input: z.union([z.string(), z.array(z.string()).min(1)]),
      dimensions: z.number().int().positive().optional()
    }
  },
  async ({ model, input, dimensions }) => {
    const embedding = await createEmbedding({
      model,
      input,
      dimensions
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            model: getModel(model),
            resolved_model: embedding.model ?? getModel(model),
            vectors: embedding.data.length,
            dimensions: embedding.data[0]?.embedding.length ?? 0
          }, null, 2)
        }
      ],
      structuredContent: {
        model: getModel(model),
        resolved_model: embedding.model ?? getModel(model),
        embeddings: embedding.data.map((item) => item.embedding),
        usage: embedding.usage ?? null
      }
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
