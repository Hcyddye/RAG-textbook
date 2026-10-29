@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   RAG 教材知识问答 - 演示服务
echo   启动后浏览器会自动打开 http://127.0.0.1:7860
echo   关闭本窗口即停止服务
echo ============================================
echo.
".venv\Scripts\python.exe" demo_app.py
echo.
echo 服务已停止，按任意键关闭窗口。
pause >nul
