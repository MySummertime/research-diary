// eslint.config.js

const eslintPluginPrettier = require('eslint-plugin-prettier');
const eslintConfigPrettier = require('eslint-config-prettier');

module.exports = [
  {
    // Configuration for JavaScript/TypeScript files
    files: ['**/*.{js,ts,jsx,tsx,mjs,cjs}'],

    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
    },

    plugins: {
      // This plugin runs Prettier as an ESLint rule
      prettier: eslintPluginPrettier,
    },

    rules: {
      // It reports Prettier-related issues as ESLint errors.
      'prettier/prettier': 'error',
    },
  },

  // This disables all of ESLint's core rules that conflict with Prettier.
  eslintConfigPrettier,
];
