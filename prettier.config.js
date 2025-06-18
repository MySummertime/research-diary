// prettier.config.js

module.exports = {
  // --- Global/Default Settings ---
  // These settings apply to all files unless a specific override matches.
  // They are ideal for files like text files, etc.
  printWidth: 100,
  endOfLine: 'lf',

  // --- Plugins ---
  plugins: [
    'prettier-plugin-toml', // Plugin for formatting TOML files
  ],

  // --- File-Specific Overrides ---
  overrides: [
    {
      // Best practices for JavaScript, TypeScript, JSX, TSX
      files: ['*.js', '*.ts', '*.jsx', '*.tsx', '*.mjs', '*.cjs'],
      options: {
        // Indentation: 2 spaces is the overwhelming standard.
        useTabs: false,
        tabWidth: 2,
        // Quotes: Single quotes are more common in JS/TS community style guides.
        singleQuote: true,
        // Semicolons: Always use them for safety and to avoid ASI bugs.
        semi: true,
        // Trailing Commas: Use them wherever possible for cleaner git diffs.
        trailingComma: 'all',
        // Spacing: Add spaces inside brackets for readability ({ foo: bar }).
        bracketSpacing: true,
        // Arrow Functions: Always add parens for consistency (e.g., (arg) => ...).
        arrowParens: 'always',
      },
    },
    {
      // Best practices for JSON files (JSON, JSON with Comments)
      // Note: Prettier is smart and will enforce double quotes for standard JSON
      // regardless of the 'singleQuote' setting.
      files: ['*.json', '*.jsonc'],
      options: {
        // Indentation: 2 spaces is the de-facto standard for JSON.
        useTabs: false,
        tabWidth: 2,
        // Trailing Commas: Not allowed in standard JSON.
        trailingComma: 'none',
        // Spacing: Consistent spacing for readability.
        bracketSpacing: true,
      },
    },
    {
      // Best practices for Markup, Config, and Text files
      files: ['*.md', '*.yml', '*.yaml', '*.toml', '*.txt'],
      options: {
        // Indentation: 2 spaces is the standard for readability in these formats.
        useTabs: false,
        tabWidth: 2,
        // Quotes: Let Prettier decide for consistency within these files.
        // Double quotes are common in TOML.
        singleQuote: false,
      },
    },
    {
      // Best practices for Styling and Markup files (CSS, SCSS, HTML)
      files: ['*.css', '*.scss', '*.html'],
      options: {
        // Indentation: 2 spaces is standard.
        useTabs: false,
        tabWidth: 2,
        // Quotes: Single quotes are common in CSS/SCSS, double in HTML attributes.
        // Prettier handles this distinction correctly.
        singleQuote: true,
      },
    },
  ],
};
