import express, { type Request, type Response } from "express";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import {
  bigintToNumber,
  createChatCompletion,
  createChatCompletionStream,
  createEmbedding,
  getJsonMode,
  getMaxCompletionTokens,
  getModel,
  getTemperature,
  groqConfig,
  listModels,
  type OllamaMessage,
  toChatMessage,
  toIso
} from "./groq.js";

type OllamaGenerateRequest = {
  model?: string;
  prompt?: string;
  system?: string;
  template?: string;
  stream?: boolean;
  raw?: boolean;
  format?: "json" | Record<string, unknown>;
  options?: Record<string, unknown>;
};

type OllamaChatRequest = {
  model?: string;
  messages?: OllamaMessage[];
  stream?: boolean;
  format?: "json" | Record<string, unknown>;
  options?: Record<string, unknown>;
};

type OllamaEmbedRequest = {
  model?: string;
  input?: string | string[];
  truncate?: boolean;
  dimensions?: number;
};

const port = Number(process.env.PORT ?? 11435);

const app = express();
app.use(express.json({ limit: "10mb" }));

app.use((_, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");

  next();
});

app.options("*", (_, res) => {
  res.sendStatus(204);
});

app.get("/", (_, res) => {
  res.json({
    ok: true,
    name: "groq-ollama-proxy",
    provider: "groq",
    base_url: groqConfig.groqBaseUrl,
    default_model: groqConfig.defaultModel
  });
});

app.get("/api/tags", async (_, res) => {
  const models = await listModels();
  res.json({
    models: models.map((model) => ({
      name: model.id,
      model: model.id,
      modified_at: toIso(model.created),
      size: 0,
      digest: `groq:${model.id}`,
      details: {
        family: groqConfig.proxyName,
        format: "remote",
        parameter_size: "unknown",
        quantization_level: "remote"
      }
    }))
  });
});

app.post("/api/show", async (req, res) => {
  const requestedModel = getModel(req.body?.name ?? req.body?.model);
  const models = await listModels();
  const matched = models.find((entry) => entry.id === requestedModel);

  res.json({
    license: "See provider terms",
    modelfile: `FROM ${requestedModel}`,
    parameters: "provider=groq",
    template: "{{ .Prompt }}",
    details: {
      parent_model: "",
      format: "remote",
      family: groqConfig.proxyName,
      families: [groqConfig.proxyName],
      parameter_size: "unknown",
      quantization_level: "remote"
    },
    model_info: matched ?? {
      id: requestedModel,
      provider: "groq"
    },
    capabilities: [
      "completion",
      "chat"
    ]
  });
});

app.post("/api/generate", asyncHandler(async (req, res) => {
  const body = req.body as OllamaGenerateRequest;
  const model = getModel(body.model);
  const stream = body.stream !== false;
  const prompt = buildPrompt(body);

  if (!prompt) {
    res.status(400).json({ error: "prompt is required" });
    return;
  }

  const messages: ChatCompletionMessageParam[] = [{ role: "user", content: prompt }];

  if (stream) {
    await streamGenerate(res, model, messages, body);
    return;
  }

  const startedAt = process.hrtime.bigint();
  const completion = await createChatCompletion({
    model,
    messages,
    temperature: getTemperature(body.options),
    maxCompletionTokens: getMaxCompletionTokens(body.options),
    jsonMode: getJsonMode(body.format)
  });
  const endedAt = process.hrtime.bigint();
  const content = completion.choices[0]?.message?.content ?? "";

  res.json({
    model: completion.model ?? model,
    created_at: new Date().toISOString(),
    response: content,
    done: true,
    done_reason: completion.choices[0]?.finish_reason ?? "stop",
    total_duration: bigintToNumber(endedAt - startedAt),
    load_duration: 0,
    prompt_eval_count: completion.usage?.prompt_tokens ?? 0,
    prompt_eval_duration: 0,
    eval_count: completion.usage?.completion_tokens ?? 0,
    eval_duration: 0
  });
}));

app.post("/api/chat", asyncHandler(async (req, res) => {
  const body = req.body as OllamaChatRequest;
  const model = getModel(body.model);
  const stream = body.stream !== false;
  const messages = (body.messages ?? []).map(toChatMessage);

  if (!messages.length) {
    res.status(400).json({ error: "messages are required" });
    return;
  }

  if (stream) {
    await streamChat(res, model, messages, body);
    return;
  }

  const startedAt = process.hrtime.bigint();
  const completion = await createChatCompletion({
    model,
    messages,
    temperature: getTemperature(body.options),
    maxCompletionTokens: getMaxCompletionTokens(body.options),
    jsonMode: getJsonMode(body.format)
  });
  const endedAt = process.hrtime.bigint();
  const content = completion.choices[0]?.message?.content ?? "";

  res.json({
    model: completion.model ?? model,
    created_at: new Date().toISOString(),
    message: {
      role: "assistant",
      content
    },
    done: true,
    done_reason: completion.choices[0]?.finish_reason ?? "stop",
    total_duration: bigintToNumber(endedAt - startedAt),
    load_duration: 0,
    prompt_eval_count: completion.usage?.prompt_tokens ?? 0,
    prompt_eval_duration: 0,
    eval_count: completion.usage?.completion_tokens ?? 0,
    eval_duration: 0
  });
}));

app.post("/api/embed", asyncHandler(async (req, res) => {
  const body = req.body as OllamaEmbedRequest;
  const model = getModel(body.model);
  const input = body.input;

  if (input === undefined) {
    res.status(400).json({ error: "input is required" });
    return;
  }

  const startedAt = process.hrtime.bigint();
  const embedding = await createEmbedding({
    model,
    input,
    dimensions: body.dimensions
  });
  const endedAt = process.hrtime.bigint();

  res.json({
    model: embedding.model ?? model,
    embeddings: embedding.data.map((item) => item.embedding),
    total_duration: bigintToNumber(endedAt - startedAt),
    load_duration: 0,
    prompt_eval_count: embedding.usage?.prompt_tokens ?? 0
  });
}));

app.use((error: unknown, _: Request, res: Response, __: express.NextFunction) => {
  const status = getErrorStatus(error);
  const message = getErrorMessage(error);

  res.status(status).json({
    error: message
  });
});

app.listen(port, () => {
  console.log(`groq-ollama-proxy listening on http://localhost:${port}`);
});

function buildPrompt(body: OllamaGenerateRequest): string {
  const segments = [];

  if (body.system) {
    segments.push(body.system);
  }

  if (body.template && !body.raw) {
    segments.push(body.template.replace("{{ .Prompt }}", body.prompt ?? ""));
  } else if (body.prompt) {
    segments.push(body.prompt);
  }

  return segments.join("\n\n").trim();
}

async function streamGenerate(
  res: Response,
  model: string,
  messages: ChatCompletionMessageParam[],
  body: OllamaGenerateRequest
): Promise<void> {
  const startedAt = process.hrtime.bigint();
  const stream = await createChatCompletionStream({
    model,
    messages,
    temperature: getTemperature(body.options),
    maxCompletionTokens: getMaxCompletionTokens(body.options),
    jsonMode: getJsonMode(body.format)
  });
  let resolvedModel = model;

  res.setHeader("Content-Type", "application/x-ndjson");

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    resolvedModel = chunk.model ?? resolvedModel;
    if (!delta) {
      continue;
    }

    res.write(`${JSON.stringify({
      model: resolvedModel,
      created_at: new Date().toISOString(),
      response: delta,
      done: false
    })}\n`);
  }

  const endedAt = process.hrtime.bigint();
  res.write(`${JSON.stringify({
    model: resolvedModel,
    created_at: new Date().toISOString(),
    response: "",
    done: true,
    done_reason: "stop",
    total_duration: bigintToNumber(endedAt - startedAt),
    load_duration: 0,
    prompt_eval_count: 0,
    prompt_eval_duration: 0,
    eval_count: 0,
    eval_duration: 0
  })}\n`);
  res.end();
}

async function streamChat(
  res: Response,
  model: string,
  messages: ChatCompletionMessageParam[],
  body: OllamaChatRequest
): Promise<void> {
  const startedAt = process.hrtime.bigint();
  const stream = await createChatCompletionStream({
    model,
    messages,
    temperature: getTemperature(body.options),
    maxCompletionTokens: getMaxCompletionTokens(body.options),
    jsonMode: getJsonMode(body.format)
  });
  let resolvedModel = model;

  res.setHeader("Content-Type", "application/x-ndjson");

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    resolvedModel = chunk.model ?? resolvedModel;
    if (!delta) {
      continue;
    }

    res.write(`${JSON.stringify({
      model: resolvedModel,
      created_at: new Date().toISOString(),
      message: {
        role: "assistant",
        content: delta
      },
      done: false
    })}\n`);
  }

  const endedAt = process.hrtime.bigint();
  res.write(`${JSON.stringify({
    model: resolvedModel,
    created_at: new Date().toISOString(),
    message: {
      role: "assistant",
      content: ""
    },
    done: true,
    done_reason: "stop",
    total_duration: bigintToNumber(endedAt - startedAt),
    load_duration: 0,
    prompt_eval_count: 0,
    prompt_eval_duration: 0,
    eval_count: 0,
    eval_duration: 0
  })}\n`);
  res.end();
}

function getErrorStatus(error: unknown): number {
  if (typeof error === "object" && error !== null && "status" in error) {
    const status = (error as { status?: unknown }).status;
    if (typeof status === "number") {
      return status;
    }
  }

  return 500;
}

function getErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string") {
      return message;
    }
  }

  return "Unknown server error";
}

function asyncHandler(
  handler: (req: Request, res: Response) => Promise<void>
): express.RequestHandler {
  return (req, res, next) => {
    handler(req, res).catch(next);
  };
}
