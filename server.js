import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import OpenAI from "openai";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const JW_PROMPT = `
You are JW Architect.
Be human, direct, grounded, and clear.
Prioritize truth, coherence, usefulness, and emotional steadiness.
Do not overexplain unless asked.
`;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function loadJson(fileName) {
  const filePath = path.join(__dirname, fileName);
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s:]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value) {
  return normalizeText(value)
    .split(" ")
    .filter((token) => token.length > 2);
}

function safeStringify(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function excerpt(value, maxLength = 500) {
  const text = typeof value === "string" ? value : safeStringify(value);
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

//
// 🔥 CIVILIZATION INDEX
//
function buildCivilizationIndex(civilizationMaster) {
  const entries = [];

  for (const deck of civilizationMaster.decks || []) {
    if (deck.teacher) {
      entries.push({
        moduleKey: "civilization_master",
        title: deck.teacher.title || `${deck.title} Teacher`,
        itemType: "teacher",
        summary: deck.teacher.summary || "",
        raw: deck.teacher
      });
    }

    for (const items of Object.values(deck.cards || {})) {
      for (const item of items || []) {
        entries.push({
          moduleKey: "civilization_master",
          title: item.title || item.id,
          itemType: item.card_type || "card",
          summary: item.summary || item.description || "",
          raw: item
        });
      }
    }
  }

  return entries.map((e) => ({
    ...e,
    searchableText: normalizeText(`${e.title} ${e.summary}`)
  }));
}

//
// 📖 SCRIPTURE INDEX
//
function buildScriptureIndex(moduleKey, data) {
  const entries = [];

  for (const [book, chapters] of Object.entries(data || {})) {
    entries.push({
      moduleKey,
      title: book,
      itemType: "book",
      summary: `Book of ${book}`,
      raw: chapters
    });

    for (const [chapter, verses] of Object.entries(chapters || {})) {
      const combined = Object.values(verses).join(" ");

      entries.push({
        moduleKey,
        title: `${book} ${chapter}`,
        itemType: "chapter",
        summary: excerpt(combined),
        raw: verses
      });

      for (const [verse, text] of Object.entries(verses || {})) {
        entries.push({
          moduleKey,
          title: `${book} ${chapter}:${verse}`,
          itemType: "verse",
          summary: text,
          raw: text
        });
      }
    }
  }

  return entries.map((e) => ({
    ...e,
    searchableText: normalizeText(`${e.title} ${e.summary}`)
  }));
}

//
// 📚 STORY INDEX (NEW)
//
function buildStoryIndex(moduleKey, data) {
  const entries = [];

  function walk(node, path = []) {
    if (!node) return;

    // String leaf
    if (typeof node === "string") {
      entries.push({
        moduleKey,
        title: path.join(" — "),
        itemType: "story_text",
        summary: node,
        raw: node
      });
      return;
    }

    // Array
    if (Array.isArray(node)) {
      node.forEach((item, i) => {
        walk(item, [...path, `item ${i + 1}`]);
      });
      return;
    }

    // Object
    if (typeof node === "object") {
      const title =
        node.title ||
        node.name ||
        node.id ||
        path[path.length - 1] ||
        "story";

      const summary =
        node.summary ||
        node.description ||
        node.text ||
        node.content ||
        "";

      if (summary) {
        entries.push({
          moduleKey,
          title: path.length ? `${path.join(" — ")} — ${title}` : title,
          itemType: "story_node",
          summary,
          raw: node
        });
      }

      for (const [key, value] of Object.entries(node)) {
        if (["summary", "description", "text", "content"].includes(key)) continue;
        walk(value, [...path, title]);
      }
    }
  }

  walk(data);

  return entries.map((e) => ({
    ...e,
    searchableText: normalizeText(`${e.title} ${e.summary}`)
  }));
}

//
// 🔧 GENERIC INDEX
//
function buildStructuredModuleIndex(moduleKey, data, fields = []) {
  return fields.map((field) => {
    const value = data?.[field];
    return {
      moduleKey,
      title: field,
      itemType: "field",
      summary: excerpt(value),
      raw: value,
      searchableText: normalizeText(`${field} ${safeStringify(value)}`)
    };
  });
}

//
// 🔍 SCORING
//
function scoreEntry(entry, tokens, raw) {
  let score = 0;
  const title = normalizeText(entry.title);

  for (const t of tokens) {
    if (title.includes(t)) score += 4;
    if (entry.searchableText.includes(t)) score += 2;
  }

  if (raw && title.includes(raw)) score += 8;
  if (entry.itemType === "verse") score += 1;

  return score;
}

function dedupe(entries) {
  const seen = new Set();
  return entries.filter((e) => {
    const key = `${e.moduleKey}:${e.title}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

//
// 🧭 ROUTING
//
function detectRoute(message, mode, config) {
  const text = normalizeText(message);
  const routes = config.route_definitions;

  if (mode && routes[mode]) {
    return { routeName: mode, routeConfig: routes[mode] };
  }

  for (const [name, route] of Object.entries(routes)) {
    if (route.keywords?.some((k) => text.includes(k))) {
      return { routeName: name, routeConfig: route };
    }
  }

  return { routeName: "default", routeConfig: routes.default };
}

//
// 📦 CONTEXT
//
function retrieve(message, modules, config, indexes) {
  const tokens = tokenize(message);
  const raw = normalizeText(message);
  const topK = config.retrieval_policy.default_top_k;

  let results = [];

  for (const key of modules) {
    const entries = indexes[key] || [];

    const ranked = entries
      .map((e) => ({ ...e, score: scoreEntry(e, tokens, raw) }))
      .filter((e) => e.score > 0)
      .sort((a, b) => b.score - a.score);

    results.push(...ranked.slice(0, topK));
  }

  return dedupe(results).slice(0, 12);
}

//
// 🚀 INIT
//
async function initialize() {
  const runtime = await loadJson("brain_core_runtime.json");
  const registry = runtime.source_registry;
  const modules = {};

  for (const [key, cfg] of Object.entries(registry)) {
    try {
      modules[key] = await loadJson(cfg.file);
    } catch (e) {
      console.warn("Load fail:", key);
    }
  }

  const indexes = {};

  if (modules.civilization_master) {
    indexes.civilization_master = buildCivilizationIndex(modules.civilization_master);
  }

  for (const [key, data] of Object.entries(modules)) {
    if (key === "civilization_master") continue;

    const role = registry[key]?.role;

    if (role === "scripture_corpus") {
      indexes[key] = buildScriptureIndex(key, data);
    } else if (role === "story_reference") {
      indexes[key] = buildStoryIndex(key, data);
    } else {
      indexes[key] = buildStructuredModuleIndex(
        key,
        data,
        registry[key]?.retrieval_fields || []
      );
    }
  }

  return { runtime, indexes };
}

const brain = await initialize();

//
// 💬 CHAT
//
app.post("/chat", async (req, res) => {
  try {
    const { message, requestedMode } = req.body;

    const route = detectRoute(message, requestedMode, brain.runtime);
    const context = retrieve(
      message,
      route.routeConfig.selected_modules,
      brain.runtime,
      brain.indexes
    );

    const response = await client.responses.create({
      model: "gpt-5.4",
      input: `${JW_PROMPT}\n\n${message}\n\nContext:\n${JSON.stringify(context).slice(0, 4000)}`
    });

    res.json({
      reply: response.output_text,
      route: route.routeName,
      sources: context.map((c) => c.title)
    });

  } catch (err) {
    res.status(500).json({ error: "Server error" });
  }
});

app.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});