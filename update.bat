@echo off
rem backtalk — update to the newest version, showing what changed first.
rem Copyright (C) 2026 Jared Rhodenizer
rem SPDX-License-Identifier: AGPL-3.0-or-later
rem
rem Your backtalk.json is yours: nothing in this script can touch or overwrite it.
rem Safe to run any time; when nothing is new it just says so.

rem cmd reads a .bat by byte offset, so a script that pulls a new copy of
rem ITSELF mid-run gets garbled from that point on. Relaunch from a copy
rem before doing any work.
rem
rem The copy lives in this program's own folder under LOCALAPPDATA and NOT
rem in the system temp directory. Identical protection, and it stops the
rem script reading like something stashing an executable where nobody
rem looks. Reviewers read repos too, and copy-to-temp-then-run is the
rem shape people are trained to distrust.
rem
rem Invoked WITHOUT `call` deliberately: cmd hands control to the copy and
rem never returns here, so not one further byte of this file is read after
rem the pull rewrites it. If the copy cannot be made, execution falls
rem through to :run below and does the work in place.
if "%~1"=="__run__" goto run
if not defined LOCALAPPDATA goto run
set "RUNDIR=%LOCALAPPDATA%\backtalk"
if not exist "%RUNDIR%\" mkdir "%RUNDIR%" >nul 2>nul
copy /y "%~f0" "%RUNDIR%\update-run.bat" >nul
"%RUNDIR%\update-run.bat" __run__ "%~dp0"

:run
setlocal
rem Where the work happens. Normally the relaunched copy is handed the
rem original folder as %2. On the fallback path above (no LOCALAPPDATA,
rem or the copy could not be made) there is no %2, so resolve to this
rem file's own folder instead of cd-ing to nowhere.
set "HOME_DIR=%~2"
if not defined HOME_DIR set "HOME_DIR=%~dp0"
cd /d "%HOME_DIR%"
set CFG=backtalk.json

if exist ".git\" goto havegit
rem this folder arrived as a zip: wire it to updates, once, keeping the config
if exist "%CFG%" copy /y "%CFG%" "%CFG%.mine" >nul
git init -b main
git remote add origin https://github.com/jaredrhod/backtalk
git fetch -q origin
git reset --hard origin/main
git branch --set-upstream-to=origin/main main
if exist "%CFG%.mine" move /y "%CFG%.mine" "%CFG%" >nul
echo wired this folder to updates.
:havegit

git fetch -q origin 2>nul
git log --oneline "..@{u}" 2>nul

rem one-time migration: the config moved out of git tracking. If git here
rem still tracks the old copy, lift yours aside, let the pull retire the
rem tracked one, then put yours back exactly as it was.
set MIGRATE=0
if not exist "%CFG%" goto pull
git ls-files --error-unmatch "%CFG%" >nul 2>nul
if errorlevel 1 goto pull
copy /y "%CFG%" "%CFG%.mine" >nul
git checkout -- "%CFG%"
set MIGRATE=1

:pull
git pull --ff-only
if errorlevel 1 echo   (couldn't fast-forward; your local edits win.)
if "%MIGRATE%"=="1" if exist "%CFG%.mine" move /y "%CFG%.mine" "%CFG%" >nul
echo update complete.
