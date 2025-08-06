# 🔐 GitHub 上传问题解决指南

## 当前问题
遇到 403 权限错误，需要重新设置 GitHub 认证。

## 解决步骤

### 1. 创建新的 Personal Access Token

1. 访问 GitHub Settings: https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置以下选项：
   - **Note**: `PrismRAG Project Upload`
   - **Expiration**: 选择合适的过期时间（建议 90 days）
   - **Scopes**: 勾选以下权限
     - ✅ `repo` (完整仓库访问权限)
     - ✅ `workflow` (如果需要 GitHub Actions)
     - ✅ `write:packages` (如果需要发布包)

4. 点击 "Generate token"
5. **重要**: 立即复制生成的 token（只显示一次）

### 2. 更新远程仓库 URL

```bash
# 将 YOUR_NEW_TOKEN 替换为刚才生成的 token
git remote set-url origin https://YOUR_NEW_TOKEN@github.com/xuanyue2019/prismrag.git
```

### 3. 推送代码

```bash
git push --set-upstream origin main
```

## 🔧 备选方案

如果 token 方式仍有问题，可以尝试：

### 方案 A: 使用 SSH 密钥
```bash
# 设置 SSH 远程地址
git remote set-url origin git@github.com:xuanyue2019/prismrag.git
git push --set-upstream origin main
```

### 方案 B: 使用 GitHub CLI
```bash
# 安装 GitHub CLI (如果未安装)
brew install gh

# 登录并推送
gh auth login
git push --set-upstream origin main
```

## 📋 当前状态检查

运行以下命令检查状态：
```bash
git remote -v
git status
git log --oneline -3
```

## 🎯 预期结果

成功后应该看到：
```
Enumerating objects: X, done.
Counting objects: 100% (X/X), done.
...
To https://github.com/xuanyue2019/prismrag.git
 * [new branch]      main -> main
Branch 'main' set up to track 'origin/main'.
```