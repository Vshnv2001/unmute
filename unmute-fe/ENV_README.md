# Environment Variables Configuration

This guide explains how to configure environment variables for the Unmute frontend.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your backend URL:**
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8000
   VITE_WS_BASE_URL=ws://127.0.0.1:8000
   ```

3. **Restart the dev server** (environment variables are loaded at build time)

## Environment Variables

### `VITE_API_BASE_URL`
- **Description:** Base URL for the backend API (HTTP/HTTPS)
- **Default:** `http://127.0.0.1:8000`
- **Examples:**
  - Local: `http://127.0.0.1:8000`
  - Docker: `http://localhost:8000`
  - Production: `https://api.yourdomain.com`

### `VITE_WS_BASE_URL`
- **Description:** Base URL for WebSocket connections
- **Default:** `ws://127.0.0.1:8000`
- **Examples:**
  - Local: `ws://127.0.0.1:8000`
  - Docker: `ws://localhost:8000`
  - Production: `wss://api.yourdomain.com`

## Configuration for Different Environments

### Local Development
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_BASE_URL=ws://127.0.0.1:8000
```

### Docker Setup
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### Production
```env
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WS_BASE_URL=wss://api.yourdomain.com
```

## Important Notes

1. **Vite Prefix Required:** All environment variables must start with `VITE_` to be exposed to the client code.

2. **Restart Required:** After changing `.env`, you must restart the Vite dev server:
   ```bash
   # Stop the server (Ctrl+C)
   # Then restart
   npm run dev
   ```

3. **Build Time:** Environment variables are embedded at build time, not runtime. For production builds, set them before building:
   ```bash
   VITE_API_BASE_URL=https://api.production.com npm run build
   ```

4. **Security:** Never commit `.env` files with sensitive data. Use `.env.example` as a template.

5. **Gitignore:** The following files should be in `.gitignore`:
   - `.env`
   - `.env.local`
   - `.env.*.local`

## File Priority

Vite loads environment files in this order (higher priority first):
1. `.env.local` (loaded for all modes except test)
2. `.env.[mode].local` (e.g., `.env.production.local`)
3. `.env.[mode]` (e.g., `.env.production`)
4. `.env`

## Troubleshooting

### Environment variables not working?
1. Verify the variable starts with `VITE_`
2. Restart the dev server
3. Check for typos in the variable name
4. Use `console.log(import.meta.env.VITE_API_BASE_URL)` to debug

### Still seeing hardcoded URLs?
- Clear Vite cache: `rm -rf node_modules/.vite`
- Rebuild: `npm run build`

### WebSocket connection failing?
- Ensure `VITE_WS_BASE_URL` uses `ws://` (not `http://`)
- For HTTPS sites, use `wss://` (not `ws://`)

## Usage in Code

```typescript
// Access environment variables
const apiUrl = import.meta.env.VITE_API_BASE_URL
const wsUrl = import.meta.env.VITE_WS_BASE_URL

// With fallback
const apiUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
```

## TypeScript Support

Add type definitions in `src/vite-env.d.ts`:

```typescript
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_WS_BASE_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
```
