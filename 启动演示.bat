@echo off
setlocal
cd /d "%~dp0"
title RAG 教材知识问答 - 演示服务

echo ============================================
echo   RAG 教材知识问答 - 本地演示服务
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到 .venv\Scripts\python.exe
    echo        这台电脑上的虚拟环境不完整，无法启动实时问答。
    echo        只想看效果的话，直接双击 RAG界面.html 即可。
    echo.
    pause
    exit /b 1
)

rem 端口已被占用说明服务已经在跑，不要再启动第二个，否则会因端口冲突一闪就退出
".venv\Scripts\python.exe" -c "import socket,sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1',7860))==0 else 1)" >nul 2>&1
if %errorlevel%==0 (
    echo 检测到服务已在运行，直接打开浏览器。
    echo 若浏览器没反应，手动访问 http://127.0.0.1:7860
    start "" "http://127.0.0.1:7860"
    timeout /t 2 >nul 2>&1
    exit /b 0
)

echo 正在启动服务，加载模型约需 10 秒，请稍等...
echo 浏览器会自动打开 http://127.0.0.1:7860
echo 关闭本窗口即停止服务。
echo.

".venv\Scripts\python.exe" demo_app.py
set "CODE=%errorlevel%"

echo.
if not "%CODE%"=="0" (
    echo [错误] 服务异常退出，退出码 %CODE%
    echo        常见原因：
    echo          1. 7860 端口被别的程序占用
    echo          2. 依赖缺失，请按 README 重新安装
    echo          3. deepseek_api_key.txt 里的 Key 无效
    echo.
)
echo 服务已停止，按任意键关闭窗口。
pause >nul
