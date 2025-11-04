#!/bin/bash

# =========================================================================
#             W&B 强制同步脚本
#
# 这个脚本会将 'wandb/' 目录下的所有本地 run 文件夹
# 强制同步到你指定的 entity (用户名) 和 project (项目) 下。
#
# 使用方法:
# 1. 登录你自己的 wandb 账号: wandb login
# 2. 修改下面的 YOUR_ENTITY 和 YOUR_PROJECT 变量。
# 3. 给脚本添加执行权限: chmod +x sync_wandb.sh
# 4. 在包含 'wandb' 文件夹的目录中运行: ./sync_wandb.sh
#
# =========================================================================


# --- !! 1. 修改这里 !! ---

# 请替换成你自己的 wandb 用户名 (或团队名)
YOUR_ENTITY="haokun-lin"

# 请替换成你想在自己账号下创建或同步到的项目名
YOUR_PROJECT="EQA_Haokun"

# --- !! 2. 修改完毕 !! ---


# 脚本开始
echo "准备同步..."
echo "目标账户 (Entity): $YOUR_ENTITY"
echo "目标项目 (Project): $YOUR_PROJECT"
echo ""

# 检查 'wandb' 目录是否存在
if [ ! -d "/vast/users/xiaodan/haokunlin/Continual_LLaVA/wandb" ]; then
  echo "错误：未在当前目录下找到 'wandb' 文件夹。"
  echo "请确保你在正确的位置运行此脚本。"
  exit 1
fi

# 检查 wandb/run-* 目录是否存在
if ! ls /vast/users/xiaodan/haokunlin/Continual_LLaVA/wandb/run-* 1> /dev/null 2>&1; then
    echo "错误：在 'wandb/' 目录中未找到任何 'run-*' 文件夹。"
    exit 1
fi

# 循环遍历所有 'wandb/run-' 开头的目录
for run_dir in /vast/users/xiaodan/haokunlin/Continual_LLaVA/wandb/run-*; do
  # 确保它是一个目录
  if [ -d "$run_dir" ]; then
    echo "======================================================"
    echo "== 正在同步: $run_dir"
    echo "======================================================"
    
    # 运行 wandb sync 命令，强制指定 project 和 entity
    wandb sync --project "$YOUR_PROJECT" --entity "$YOUR_ENTITY" "$run_dir"
    
    echo "== 同步完成: $run_dir"
    echo ""
  else
    echo "跳过: $run_dir (这不是一个目录)"
  fi
done

echo "------------------------------------------------------"
echo "  所有本地 run 均已尝试同步！"
echo "------------------------------------------------------"