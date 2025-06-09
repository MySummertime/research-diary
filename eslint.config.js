// eslint.config.js
const eslintPluginPrettier = require('eslint-plugin-prettier');

module.exports = [
  {
    files: ['**/*.{js,ts,jsx,tsx}'],

    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
    },

    rules: {
      'prettier/prettier': 'error', // 违反 Prettier 规则就报错
      semi: ['error', 'always'], // 强制加分号
      quotes: ['error', 'single'], // 强制用单引号
      indent: 'off', // 关闭 ESLint 的缩进检查
    },

    plugins: {
      prettier: eslintPluginPrettier,
    },
  },
];
