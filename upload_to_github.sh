#!/bin/bash

# GitHub 上传脚本
# 用户名: xuanyue2019

echo "🚀 准备上传 PrismRAG 项目到 GitHub..."

# 设置用户名
username="xuanyue2019"

# 检查是否已经设置了远程仓库
if git remote get-url origin 2>/dev/null; then
    echo "✅ 远程仓库已设置"
    current_origin=$(git remote get-url origin)
    echo "当前远程仓库: $current_origin"
else
    echo "🔗 添加远程仓库..."
    git remote add origin https://github.com/$username/prismrag.git
    echo "已添加远程仓库: https://github.com/$username/prismrag.git"
fi

# 推送到 GitHub
echo "📤 推送代码到 GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "🎉 成功上传到 GitHub!"
    echo "📋 项目地址: https://github.com/$username/prismrag"
    echo ""
    echo "🔧 接下来你可以:"
    echo "1. 在 GitHub 上查看你的项目"
    echo "2. 设置 GitHub Actions (已自动配置)"
    echo "3. 邀请协作者"
    echo "4. 创建 Issues 和 Pull Requests"
    echo ""
    echo "📚 更多信息请查看:"
    echo "- README.md: 项目介绍和使用指南"
    echo "- docs/: 详细文档"
    echo "- CONTRIBUTING.md: 贡献指南"
else
    echo "❌ 上传失败，请检查:"
    echo "1. GitHub 用户名是否正确"
    echo "2. 是否有仓库访问权限"
    echo "3. 网络连接是否正常"
fi