#!/bin/bash
command=$1
max_attempts=10
wait_time=30
attempt=0

while [ $attempt -lt $max_attempts ]
do
    if eval $command;then
        echo "Command executed successfully"
        break
    else
        # 如果命令执行失败，则增加计数器并等待一段时间后重试
        attempt=$((attempt+1))
        echo "Command failed $attempt, Retrying in $wait_time seconds..."
        sleep $wait_time
    fi
done
if [ $attempt -eq $max_attempts ];then
    echo "Command failed after $max_attempts attempts."
    exit 1
fi
