# 如何设置 Xi-Learning 文档站 (GitHub Pages)

资料生成器 GitHub Pages 文档集已成功生成，位于 `docs/` 目录。

请按照以下步骤配置 GitHub Pages 托管：

## 📋 步骤清单

### 1. 在 GitHub 上创建仓库并初始化

```bash
# 在 GitHub 上创建新仓库（例如 xi-learning）
# 然后在本地克隆

git clone https://github.com/xi-learning/xi-learning.git
cd xi-learning

# 添加代码文件
git add .
git commit -m "Initial commit: Xi-Learning framework"
git push origin main
```

### 2. 配置 GitHub Pages

1. 进入 GitHub 仓库页面：`https://github.com/xi-learning/xi-learning`
2. 点击 **Settings** → **Pages**
3. 在 **Source** 下，选择 **Deploy from a branch**
4. 分支选择：`main` → `/docs` 目录
5. 点击 **Save**

### 3. 验证站点

GitHub 会为您生成一个网址：`https://xe-learning.github.io/xi-learning/`

访问该链接应能看到文档主页 HTML（已被重新命名为 README.md）。

## 🗓️ 目录结构

```
xi-learning/
├── docs/
│   ├── index.html              # 站点主页
│   ├── about/
│   │   └── introduction.html   # 项目入门介绍
│   ├── theory/
│   │   ├── axioms.html         # 16 条公理 +9 大定理
│   │   └── axioms.html
│   ├── implementations/
│   │   └── code.html          # 代码实现说明
│   ├── publications/
│   │   └── paper.html         # 论文的 abstract + 投稿
│   ├── contributing/
│   │   └── get-started.html   # 贡献指南
│   ├── resources/
│   │   └── guides.html        # 学习资源与教程
│   └── styles/
│       └── main.css           # 统一样式表
├── .github/
│   └── pages/
│       └── .htaccess          # URL 重写规则
└── README.md                  # 仓库说明文档
```

## 🎯 核心功能

- ✅ **单页应用体验**：所有页面可以无缝导航
- ✅ **响应式设计**：桌面和移动端友好
- ✅ **数学模式**：支持 LaTeX 风格数学公式
- ✅ **权威导航**：清晰的大纲和专题页面
- ✅ **高质量文档**：包含 README.md, CONTRIBUTING.md, SETUP.md

## 🚀 下一步

### 部署后检查
1. 访问 `https://xe-learning.github.io/xi-learning/docs/index.html`
2. 测试每个导航菜单项
3. 验证代码片段正确显示
4. 检查链接和引用

### 后续工作
- 从 `xi-learning.tex` 生成正式 PDF
- 上传到 arXiv 并获得出版号
- 在 GitHub 上设置 CI/CD 自动化测试
- 添加自定义域名（可选）

## 📁 相关资源

- **项目根目录**：[README.md](https://github.com/xi-learning/xi-learning/blob/main/README.md)
- **代码仓库**：[GitHub Repository](https://github.com/xi-learning/xi-learning)
- **学术论文**：arXiv:2400.00000 (待提交)
- **License**：MIT (代码) + CC-BY-SA 4.0 (理论)

## ❓ 常见问题

**Q: 为什么我没有看到页面？**
A: GitHub Pages 可能需要 1-2 分钟才能生成。如果长时间没有响应，请检查上述步骤。

**Q: 如何自定义自定义域名？**
A: 在 Settings → Pages 中选择 `Custom Domain`，输入您的域名，然后在您的 DNS 配置中添加 CNAME 记录。

**Q: 文档网站如何更新？**
A: 直接在 `docs/` 目录中修改 HTML 文件，然后提交到 GitHub 仓库即可。

## 🎊 获得

Xi-Learning 文档网站现已准备就绪！您现在可以通过 GitHub Pages 向全世界展示这一理论框架。

祝你在 Xi-Learning 研究中一切顺利！🚀
