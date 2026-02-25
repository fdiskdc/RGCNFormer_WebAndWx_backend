#!/bin/bash
###
 # @Author: Chao Deng && chaodeng987@outlook.com
 # @Date: 2026-02-25 03:34:03
 # @LastEditors: Chao Deng && chaodeng987@outlook.com
 # @LastEditTime: 2026-02-25 03:35:35
 # @FilePath: /backend/run_docker.sh
 # @Description: 
 # 那只是一场游戏一场梦
 #  
 # https://orcid.org/0009-0009-8520-1656
 # DOI: 10.3390/app15158626
 # DOI: 10.3390/rs17142354
 # Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
### 

# 检查变量是否存在，如果不存在则要求输入
if [ -z "$WX_APPID" ]; then
    read -p "请输入 WX_APPID (默认 wxcb604a93d537d38a): " WX_APPID
    export WX_APPID=${WX_APPID:-wxcb604a93d537d38a}
fi

if [ -z "$WX_SECRET" ]; then
    read -p "请输入 WX_SECRET (默认 73e7bea0faafabef192ab5310688f0a2): " WX_SECRET
    export WX_SECRET=${WX_SECRET:-73e7bea0faafabef192ab5310688f0a2}
fi

# 启动容器
echo "🚀 正在使用指定的微信参数启动服务..."
docker compose up -d --build