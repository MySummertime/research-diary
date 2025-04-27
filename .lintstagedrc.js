module.exports = {
  '**/*.{js,ts,jsx,tsx}': ['eslint --fix', 'prettier --write'],
  '**/*.{md,json,yml,yaml,css,scss}': ['prettier --write'],
};
