import "dotenv/config";
import OpenAI from "openai";
import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam
} from "openai/resources/chat/completions";
import type { EmbeddingCreateParams, CreateEmbeddingResponse } from "openai/resources/embeddings";

export type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  images?: string[];
};

export type GroqModel = {
  id: string;
  owned_by?: string;
  created?: number;
};

export type ChatRequestOptions = {
  model?: string;
  messages: ChatCompletionMessageParam[];
  temperature?: number;
  maxCompletionTokens?: number;
  jsonMode?: boolean;
};

export type EmbedRequestOptions = {
  model?: string;
  input: string | string[];
  dimensions?: number;
};

export const groqConfig = {
  defaultModel: process.env.DEFAULT_MODEL ?? "openai/gpt-oss-20b",
  fallbackModels: parseModelList(process.env.FALLBACK_MODELS),
  proxyName: process.env.OLLAMA_PROXY_NAME ?? "groq",
  groqBaseUrl: process.env.GROQ_BASE_URL ?? "https://api.groq.com/openai/v1"
};

const groqApiKey = process.env.GROQ_API_KEY;

if (!groqApiKey) {
  throw new Error("GROQ_API_KEY is required.");
}

export const groqClient = new OpenAI({
  apiKey: groqApiKey,
  baseURL: groqConfig.groqBaseUrl
});

export async function listModels(): Promise<GroqModel[]> {
  try {
    const response = await groqClient.models.list();
    return response.data.map((model) => ({
      id: model.id,
      owned_by: "groq",
      created: typeof model.created === "number" ? model.created : undefined
    }));
  } catch {
    return [
      {
        id: groqConfig.defaultModel,
        owned_by: "groq"
      }
    ];
  }
}

export function getModel(model?: string): string {
  return model?.trim() || groqConfig.defaultModel;
}

export function normalizeContent(message: OllamaMessage): string {
  const text = message.content ?? "";

  if (!message.images?.length) {
    return text;
  }

  return `${text}\n\n[${message.images.length} image(s) omitted by proxy]`;
}

export function toChatMessage(message: OllamaMessage): ChatCompletionMessageParam {
  const content = normalizeContent(message);

  if (message.role === "system") {
    return { role: "system", content };
  }

  if (message.role === "assistant") {
    return { role: "assistant", content };
  }

  if (message.role === "tool") {
    return { role: "user", content: `[tool]\n${content}` };
  }

  return { role: "user", content };
}

export async function createChatCompletion(
  options: ChatRequestOptions
): Promise<ChatCompletion> {
  const { result } = await tryModels(async (model) => {
    const request: ChatCompletionCreateParamsNonStreaming = {
      model,
      messages: options.messages,
      temperature: options.temperature,
      max_completion_tokens: options.maxCompletionTokens,
      response_format: options.jsonMode ? { type: "json_object" } : undefined
    };

    return groqClient.chat.completions.create(request);
  }, options.model);

  return result;
}

export async function createChatCompletionStream(options: ChatRequestOptions) {
  const { result } = await tryModels(async (model) => {
    const request: ChatCompletionCreateParamsStreaming = {
      model,
      messages: options.messages,
      temperature: options.temperature,
      max_completion_tokens: options.maxCompletionTokens,
      response_format: options.jsonMode ? { type: "json_object" } : undefined,
      stream: true
    };

    return groqClient.chat.completions.create(request);
  }, options.model);

  return result;
}

export async function createEmbedding(
  options: EmbedRequestOptions
): Promise<CreateEmbeddingResponse> {
  const { result } = await tryModels(async (model) => {
    const request: EmbeddingCreateParams = {
      model,
      input: options.input,
      dimensions: options.dimensions
    };

    return groqClient.embeddings.create(request);
  }, options.model);

  return result;
}

export function getTemperature(options?: Record<string, unknown>): number | undefined {
  const temperature = options?.temperature;
  if (typeof temperature !== "number") {
    return undefined;
  }

  return temperature === 0 ? 1e-8 : temperature;
}

export function getMaxCompletionTokens(options?: Record<string, unknown>): number | undefined {
  const numPredict = options?.num_predict;
  return typeof numPredict === "number" ? numPredict : undefined;
}

export function getJsonMode(
  format?: "json" | Record<string, unknown>
): boolean {
  return format === "json";
}

export function toIso(unixTimestamp?: number): string {
  if (!unixTimestamp) {
    return new Date().toISOString();
  }

  return new Date(unixTimestamp * 1000).toISOString();
}

export function bigintToNumber(value: bigint): number {
  const capped = value > BigInt(Number.MAX_SAFE_INTEGER)
    ? BigInt(Number.MAX_SAFE_INTEGER)
    : value;

  return Number(capped);
}

type FallbackExecutionResult<T> = {
  model: string;
  result: T;
};

async function tryModels<T>(
  execute: (model: string) => Promise<T>,
  requestedModel?: string
): Promise<FallbackExecutionResult<T>> {
  const models = resolveModelCandidates(requestedModel);
  let lastError: unknown = undefined;

  for (const model of models) {
    try {
      const result = await execute(model);
      attachResolvedModel(result, model);
      return { model, result };
    } catch (error) {
      lastError = error;
      if (!shouldFallback(error)) {
        throw error;
      }
    }
  }

  throw lastError;
}

function resolveModelCandidates(requestedModel?: string): string[] {
  const preferred = getModel(requestedModel);
  return uniqueModels([preferred, ...groqConfig.fallbackModels.filter((model) => model !== preferred)]);
}

function uniqueModels(models: string[]): string[] {
  return [...new Set(models.map((model) => model.trim()).filter(Boolean))];
}

function parseModelList(value?: string): string[] {
  if (!value) {
    return [
      "openai/gpt-oss-20b",
      "openai/gpt-oss-120b",
      "llama-3.1-8b-instant",
      "llama-3.3-70b-versatile",
      "meta-llama/llama-4-scout-17b-16e-instruct",
      "qwen/qwen3-32b",
      "groq/compound-mini",
      "groq/compound"
    ];
  }

  return uniqueModels(value.split(","));
}

function shouldFallback(error: unknown): boolean {
  const status = getErrorStatus(error);
  const message = getErrorMessageText(error).toLowerCase();

  if (status === 429) {
    return true;
  }

  if (typeof status === "number" && status >= 500) {
    return true;
  }

  if (status === 408 || status === 409) {
    return true;
  }

  return [
    "rate limit",
    "quota",
    "capacity",
    "temporarily unavailable",
    "overloaded",
    "timeout",
    "timed out",
    "model is not available",
    "model not available",
    "does not exist",
    "engine not found",
    "service unavailable"
  ].some((needle) => message.includes(needle));
}

function attachResolvedModel(result: unknown, model: string): void {
  if (!result || typeof result !== "object") {
    return;
  }

  const target = result as Record<string, unknown>;
  if (typeof target.model !== "string" || !target.model) {
    target.model = model;
  }
}

function getErrorStatus(error: unknown): number | undefined {
  if (typeof error === "object" && error !== null && "status" in error) {
    const status = (error as { status?: unknown }).status;
    if (typeof status === "number") {
      return status;
    }
  }

  return undefined;
}

function getErrorMessageText(error: unknown): string {
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }

  return "";
}
