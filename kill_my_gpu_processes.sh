#!/bin/bash

# 查询并杀死当前用户在GPU上运行的进程

echo "========================================="
echo "查询当前用户的GPU进程..."
echo "========================================="

# 获取当前用户名
CURRENT_USER=$(whoami)

# 使用nvidia-smi查询GPU进程，过滤当前用户的进程
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

if [ -z "$GPU_PIDS" ]; then
    echo "未找到任何GPU进程"
    exit 0
fi

# 存储要杀死的进程ID
PIDS_TO_KILL=()

echo "检查进程所有者..."
echo ""

for pid in $GPU_PIDS; do
    # 检查进程是否存在
    if [ ! -d "/proc/$pid" ]; then
        continue
    fi
    
    # 获取进程所有者
    OWNER=$(ps -o user= -p $pid 2>/dev/null)
    
    if [ "$OWNER" = "$CURRENT_USER" ]; then
        # 获取进程命令
        CMD=$(ps -o cmd= -p $pid 2>/dev/null)
        echo "找到进程: PID=$pid"
        echo "  命令: $CMD"
        echo ""
        PIDS_TO_KILL+=($pid)
    fi
done

# 如果没有找到当前用户的进程
if [ ${#PIDS_TO_KILL[@]} -eq 0 ]; then
    echo "未找到当前用户 ($CURRENT_USER) 的GPU进程"
    exit 0
fi

# 显示汇总
echo "========================================="
echo "找到 ${#PIDS_TO_KILL[@]} 个属于用户 $CURRENT_USER 的GPU进程"
echo "========================================="

# 询问是否杀死
read -p "是否杀死这些进程? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "正在杀死进程..."
    for pid in "${PIDS_TO_KILL[@]}"; do
        echo "  杀死 PID $pid..."
        kill -9 $pid 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "    ✓ 成功"
        else
            echo "    ✗ 失败 (可能已经结束)"
        fi
    done
    echo ""
    echo "完成！"
    
    # 等待一下再显示当前状态
    sleep 1
    echo ""
    echo "当前GPU状态:"
    nvidia-smi
else
    echo "操作已取消"
fi


