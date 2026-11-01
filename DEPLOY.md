# 云服务器部署指南

把问答服务部署到云服务器后，手机在任何地方（有网即可）打开就能用，不再依赖家里电脑。

## 0. 准备一台服务器

推荐 **腾讯云 / 阿里云「轻量应用服务器」**，最低配即可：

- 系统：**Ubuntu 22.04**
- 配置：2 核 2G（最低配足够，本服务 CPU/内存占用都很小）
- 带宽：按量或固定 1M 都行（页面是纯文本，流量很小）

买好后记下两样东西：**公网 IP** 和 **root 密码**（或 SSH 密钥）。

## 1. 上传代码 + 数据（本地执行）

> 本地用 Git Bash 执行。下面的 `服务器IP` 换成你的公网 IP。

打包整个项目（排除 .env 和缓存），上传到服务器：

```bash
cd /c/Users/x/rag-eval
tar --exclude='.venv' --exclude='.env' --exclude='__pycache__' \
    --exclude='.pytest_cache' -czf rag-eval.tar.gz .

scp rag-eval.tar.gz root@服务器IP:/opt/
```

> 因为 `data/chunks` 里的切分结果和向量（25MB）已包含在包内，服务器**无需重新下载语料、无需重新做 embedding**。

## 2. 解压并安装（服务器执行）

登录服务器：

```bash
ssh root@服务器IP
```

解压、建虚拟环境、装依赖、配置：

```bash
mkdir -p /opt/rag-eval
tar -xzf /opt/rag-eval.tar.gz -C /opt/rag-eval
cd /opt/rag-eval

# 装 Python 依赖（Ubuntu 首次）
apt update && apt install -y python3 python3-venv python3-pip

# 填 API key
cp .env.example .env
nano .env          # 把 OPENAI_API_KEY 填成你自己的 key，保存
```

## 3. 一键部署（服务器执行）

```bash
bash deploy/setup.sh
```

脚本会自动：创建 venv → 装依赖 → 校验 .env → 安装 systemd 服务并启动。

看到 `==> Deploy done. Visit: http://...` 即成功。

## 4. 放行 8000 端口（云控制台）

这一步在**云服务商控制台**做，不是命令行：

1. 找到你的服务器 → **防火墙 / 安全组** 设置
2. 添加入站规则：协议 **TCP**，端口 **8000**，来源 **0.0.0.0/0**（允许所有）

> 忘记这步的话，手机能连服务器但打不开 8000 端口，是最常见的坑。

## 5. 手机访问

手机浏览器打开：

```
http://服务器公网IP:8000
```

即可。桌面浏览器、平板同样适用。

## 常用运维命令

```bash
systemctl status rag-qa        # 看服务状态
systemctl restart rag-qa       # 重启
journalctl -u rag-qa -f        # 实时看日志
```

## 更新代码后

本地重新打包上传、解压覆盖，再重启：

```bash
# 服务器
cd /opt/rag-eval
tar -xzf /opt/rag-eval.tar.gz -C /opt/rag-eval
systemctl restart rag-qa
```

## 说明

- 用 `http://IP:8000` 直连**不需要域名备案**；想要 https 域名才需备案 + 配 Nginx/证书。
- 本服务默认绑定 `0.0.0.0`，端口 8000。生产环境建议再套一层 Nginx 反代或换端口。
