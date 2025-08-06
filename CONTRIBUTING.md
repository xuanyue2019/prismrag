# Contributing to PrismRAG

感谢您对 PrismRAG 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请：

1. 检查 [Issues](https://github.com/xuanyue2019/prismrag/issues) 确保问题尚未被报告
2. 创建新的 Issue，包含：
   - 清晰的标题和描述
   - 重现步骤（如果是 bug）
   - 期望的行为
   - 实际的行为
   - 环境信息（Python 版本、操作系统等）

### 提交代码

1. **Fork 项目**
   ```bash
   # 在 GitHub 上 fork 项目，然后克隆到本地
   git clone https://github.com/xuanyue2019/prismrag.git
   cd prismrag
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **设置开发环境**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **进行更改**
   - 遵循现有的代码风格
   - 添加必要的测试
   - 更新文档（如果需要）

5. **运行测试**
   ```bash
   python -m pytest tests/
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   # 或
   git commit -m "fix: fix your bug description"
   ```

7. **推送并创建 Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在 GitHub 上创建 Pull Request。

## 代码规范

### Python 代码风格

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用 4 个空格缩进
- 行长度限制为 88 字符
- 使用有意义的变量和函数名

### 提交信息格式

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型包括：
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 添加测试
- `chore`: 构建过程或辅助工具的变动

### 文档

- 所有公共函数和类都应该有 docstring
- 使用 Google 风格的 docstring
- 更新相关的 README 和 API 文档

## 测试

- 为新功能添加单元测试
- 确保所有测试通过
- 测试覆盖率应该保持在合理水平

## 发布流程

项目维护者会处理版本发布：

1. 更新版本号
2. 更新 CHANGELOG
3. 创建 GitHub Release
4. 发布到 PyPI（如果适用）

## 行为准则

请遵循我们的行为准则：

- 尊重所有参与者
- 建设性地提供反馈
- 专注于对项目最有利的事情
- 展现同理心

## 获得帮助

如果您需要帮助：

1. 查看 [文档](docs/)
2. 搜索现有的 [Issues](https://github.com/xuanyue2019/prismrag/issues)
3. 创建新的 Issue 寻求帮助

感谢您的贡献！