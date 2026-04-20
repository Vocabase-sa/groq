# Groq Ollama Proxy + MCP

Serveur local compatible avec une partie de l'API Ollama, mais adossé à Groq.

Le but est de faire croire à des clients qui savent parler à Ollama qu'ils parlent à un backend local, alors que les générations partent vers `https://api.groq.com/openai/v1`.

Le dépôt contient aussi un vrai serveur MCP `stdio` pour que Codex puisse appeler Groq directement via MCP, sans passer par la compatibilité Ollama.

## Endpoints pris en charge

- `GET /`
- `GET /api/tags`
- `POST /api/show`
- `POST /api/generate`
- `POST /api/chat`
- `POST /api/embed`

## Tools MCP pris en charge

- `list_models`
- `generate_text`
- `chat_completion`
- `embed_text`

## Limites importantes

- Ce projet n'émule pas tout Ollama.
- Il ne gère pas les modèles locaux, pulls, pushes, blobs, quantization, ni la gestion de runtime Ollama.
- Les endpoints `generate` et `chat` visent la compatibilité pratique. Certains champs Ollama sont ignorés si Groq n'a pas d'équivalent direct.
- `embed` dépend du support embeddings côté Groq pour le modèle choisi. Si le modèle ou le compte ne le supporte pas, l'appel échouera côté fournisseur.
- Le serveur MCP expose des tools pratiques pour Codex, pas une surface Groq exhaustive.

## Lancement

1. Installer les dépendances:

```bash
npm install
```

2. Configurer l'environnement:

```bash
copy .env.example .env
```

3. Renseigner `GROQ_API_KEY` dans `.env`.

Tu peux aussi définir `FALLBACK_MODELS` comme une liste séparée par des virgules. Si le modèle demandé échoue pour cause d'indisponibilité fournisseur, rate limit, quota ou erreur serveur, le proxy essaiera le modèle suivant.

4. Lancer en développement:

```bash
npm run dev
```

5. Ou build + run:

```bash
npm run build
npm start
```

Par défaut, le serveur écoute sur `http://localhost:11435`, pour éviter le conflit avec Ollama.

## Lancer le serveur MCP

En développement:

```bash
npm run dev:mcp
```

Après build:

```bash
npm run build
npm run mcp
```

## Brancher le serveur MCP dans Codex

Le plus simple est d'ajouter un serveur MCP local `stdio` pointant vers le script buildé.

Exemple de bloc `~/.codex/config.toml`:

```toml
[mcp_servers.groq]
command = "node"
args = ["C:/Users/FrédéricJamoulle/Claude/tests/groq/dist/mcp.js"]
```

Puis vérifier la configuration:

```bash
codex mcp list
```

Une fois connecté, Codex pourra appeler les tools `list_models`, `generate_text`, `chat_completion` et `embed_text`.

## Brancher le serveur MCP dans Claude Code

Ajouter le serveur MCP au scope utilisateur via la CLI `claude` :

```bash
claude mcp add groq --scope user -e GROQ_API_KEY=clef_groq -- node "chemin/vers/dist/mcp.js"
```

Exemple concret sous Windows :

```bash
claude mcp add groq --scope user -e GROQ_API_KEY=gsk_xxxxxxxxxxxx -- node "C:/Users/FrédéricJamoulle/Claude/mcp/groq/dist/mcp.js"
```

Vérifier l'enregistrement :

```bash
claude mcp list
```

Claude Code pourra alors appeler les tools `list_models`, `generate_text`, `chat_completion` et `embed_text` via le préfixe `mcp__groq__`.

## Fallback de modèles

Le serveur essaie d'abord le modèle demandé. En cas d'échec temporaire côté fournisseur, il bascule automatiquement vers les modèles de `FALLBACK_MODELS`.

Cas de fallback pris en compte:

- `429` rate limit / quota
- erreurs `5xx`
- timeouts et indisponibilités temporaires
- modèle indisponible côté fournisseur

Cas non basculés:

- requête invalide côté client
- paramètres non supportés
- payload mal formé

## Exemple

```bash
curl http://localhost:11435/api/generate -d "{\"model\":\"openai/gpt-oss-20b\",\"prompt\":\"Dis bonjour\",\"stream\":false}" -H "Content-Type: application/json"
```
