module.exports = {
  types: [
    { value: 'feat', name: 'feat: 🚀 A new feature' },
    { value: 'fix', name: 'fix: 🐛 A bug fix' },
    { value: 'docs', name: 'docs: 📝 Documentation only changes' },
    { value: 'style', name: 'style: 💅 Code style changes (formatting, etc.)' },
    { value: 'refactor', name: 'refactor: 💡 Code refactoring' },
    { value: 'perf', name: 'perf: ⚡ Performance improvements' },
    { value: 'test', name: 'test: ✅ Adding or updating tests' },
    { value: 'chore', name: 'chore: 🛠 Routine task, maintenance' },
    { value: 'build', name: 'build: 🔧 Changes that affect the build system' },
    { value: 'ci', name: 'ci: 🔧 Changes to CI/CD configuration' },
  ],
  messages: {
    type: '请选择提交的类型:',
    subject: '简要描述：',
    body: '详细描述 (可选):',
    footer: '相关 issue (可选):',
    confirmCommit: '确认提交?',
  },
  skipQuestions: ['body', 'footer'],
  subjectLimit: 100, // 限制提交简述最大字符数
};
