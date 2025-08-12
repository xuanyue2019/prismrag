# 🔐 GitHub Personal Access Token 权限设置指南

## 问题诊断
当前 token 可能缺少必要的 `repo` 权限，导致推送失败。

## 解决方案：创建新的 Token

### 1. 访问 GitHub Token 设置页面
🔗 https://github.com/settings/tokens/new

### 2. 配置 Token 设置
- **Note**: `PrismRAG Project - Full Repo Access`
- **Expiration**: 选择 `90 days` 或 `No expiration`（根据需要）

### 3. 必须勾选的权限范围 (Scopes)
✅ **repo** - 完整仓库访问权限
  - ✅ repo:status - 访问提交状态
  - ✅ repo_deployment - 访问部署状态
  - ✅ public_repo - 访问公共仓库
  - ✅ repo:invite - 访问仓库邀请

可选权限（推荐）：
✅ **workflow** - 更新 GitHub Actions 工作流
✅ **write:packages** - 上传包到 GitHub Packages

### 4. 生成并复制 Token
1. 点击 "Generate token"
2. **立即复制** 生成的 token（只显示一次！）
3. 保存到安全的地方

### 5. 使用新 Token 推送代码

```bash
# 进入项目目录
cd .seal/prismRAG

# 使用新 token 更新远程 URL（替换 YOUR_NEW_TOKEN）
git remote set-url origin https://YOUR_NEW_TOKEN@github.com/xuanyue2019/prismrag.git

# 推送代码
git push origin main
```

## 🔍 验证 Token 权限

使用以下命令验证 token 是否有正确权限：

```bash
# 检查用户信息
curl -H "Authorization: token YOUR_NEW_TOKEN" https://api.github.com/user

# 检查仓库权限
curl -H "Authorization: token YOUR_NEW_TOKEN" https://api.github.com/repos/xuanyue2019/prismrag
```

## 📋 当前项目状态

你的 PrismRAG 项目包含：
- ✅ 完整的源代码实现
- ✅ 文档和配置文件
- ✅ 测试文件
- ✅ 本地 Git 提交历史
- ⏳ 等待推送到 GitHub

## 🚨 重要提醒

1. **Token 安全**: 不要在代码中硬编码 token
2. **权限最小化**: 只给予必要的权限
3. **定期更新**: 建议定期更换 token
4. **备份**: 保存 token 到安全的密码管理器

## 🎯 预期结果

成功后你应该看到：
```
Enumerating objects: X, done.
Counting objects: 100% (X/X), done.
...
To https://github.com/xuanyue2019/prismrag.git
   xxxxx..xxxxx  main -> main
```

然后你就可以访问 https://github.com/xuanyue2019/prismrag 查看你的项目了！