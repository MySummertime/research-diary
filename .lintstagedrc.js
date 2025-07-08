// .lintstagedrc.js
// This file is used by lint-staged to run scripts on staged area before committing.

module.exports = {
  // --- Global Rules for Non-Python Files ---
  // Run Prettier the Formatter to check for code style.
  '**/*.{js,jsx,ts,tsx,mjs,cjs,md,json,jsonc,yml,yaml,toml,css,scss,html}': ['prettier --write'],

  // --- JS/TS Specific Linting ---
  // Run ESLint the Linter to check for code quality.
  '**/*.{js,jsx,ts,tsx,mjs,cjs}': ['eslint --fix'],

  // --- Python Project 1: Risk_Assessment_Models ---
  // This glob specifically targets only the Python files within this project.
  'projects/risk-assessment-models/backend/**/*.py': (filenames) => {
    const projectPath = 'projects/risk-assessment-models/backend';
    const config = {
      condaEnv: `${projectPath}/envs`,
      pyprojectToml: `${projectPath}/pyproject.toml`,
    };

    const filesToProcess = filenames.join(' ');

    return [
      // Step 1: Format the code first using Ruff's formatter.
      `conda run -p ${config.condaEnv} ruff format ${filesToProcess}`,
      // Step 2: Lint the newly formatted code and autofix safe issues.
      `conda run -p ${config.condaEnv} ruff check --fix --config ${config.pyprojectToml} ${filesToProcess}`,
      // Step 3: Perform deep type checking with MyPy as the final quality gate.
      // `conda run -p ${config.condaEnv} mypy --config-file=${config.pyprojectToml} ${filesToProcess}`,
    ];
  },

  // --- Python Project 2 (Placeholder) ---
};
