import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // The dashboard consumes dynamic WebSocket/API payloads. Keep explicit
      // `any` visible in lint output, but don't fail the whole sweep on payload
      // typing debt while runtime-oriented rules stay strict.
      '@typescript-eslint/no-explicit-any': 'warn',
      // This project intentionally colocates small hooks with their provider
      // components in a few files.
      'react-refresh/only-export-components': 'warn',
      // Project/session changes intentionally reset local UI state inside
      // effects after WebSocket boundaries change.
      'react-hooks/set-state-in-effect': 'warn',
    },
  },
])
