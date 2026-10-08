@echo off
rem backtalk: talk to your Claude Code agent out loud.
rem Copyright (C) 2026 Jared Rhodenizer
rem
rem This program is free software: you can redistribute it and/or modify
rem it under the terms of the GNU Affero General Public License as published
rem by the Free Software Foundation, either version 3 of the License, or
rem (at your option) any later version.
rem
rem This program is distributed in the hope that it will be useful,
rem but WITHOUT ANY WARRANTY; without even the implied warranty of
rem MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
rem GNU Affero General Public License for more details.
rem
rem You should have received a copy of the GNU Affero General Public License
rem along with this program. If not, see <https://www.gnu.org/licenses/>.
rem
rem SPDX-License-Identifier: AGPL-3.0-or-later
rem backtalk launcher (Windows), standalone (no face/hands). Added 2026-09-26
rem so the voice line can be pointed at from a Windows Startup shortcut
rem without going through fullstack-agent\start.bat.
rem   run.bat        start a spoken conversation with your agent
cd /d "%~dp0"

echo   voice: checking packages. The FIRST run downloads a few hundred MB
echo          and can take several minutes. It is not stuck.
uv sync --inexact
if errorlevel 1 (
  echo.
  echo   The voice line's packages could not be installed, so it never
  echo   started. The reason is in the output above.
  echo.
  echo   This happened during setup, BEFORE the voice ran, so there is
  echo   nothing about it in backtalk\logs\backtalk.log.
  echo.
  pause
  exit /b 1
)

uv run python -m backtalk.main
if errorlevel 1 (
  echo.
  echo   The voice line stopped with an error. The message is above.
  echo   The log lives in backtalk\logs\backtalk.log
  pause
)
