@rem Copyright 2024 Google LLC
@rem
@rem Licensed under the Apache License, Version 2.0 (the "License");
@rem you may not use this file except in compliance with the License.
@rem You may obtain a copy of the License at
@rem
@rem     http://www.apache.org/licenses/LICENSE-2.0
@rem
@rem Unless required by applicable law or agreed to in writing, software
@rem distributed under the License is distributed on an "AS IS" BASIS,
@rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@rem See the License for the specific language governing permissions and
@rem limitations under the License.

@rem This test file runs for one Python version at a time, and is intended to
@rem be called from within the build loop.

:; Change directory to repo root.
SET script_dir="%~dp0"
cd "%~dp0"\..\..

set PYTHON_VERSION=%1
if "%PYTHON_VERSION%"=="" (
  echo "Python version was not provided, using Python 3.10"
  set PYTHON_VERSION=3.10
)

py -%PYTHON_VERSION%-64 -m venv venv
call .\venv\Scripts\activate.bat

python -m pip install --no-index --find-links=wheels bigframes --force-reinstall || goto :error

python -m pip install pytest || goto :error
python -m pytest tests/unit || goto :error
python -m pytest tests/system/small || goto :error

:; https://stackoverflow.com/a/46813196/101923
:; exit 0
exit /b 0

:error
exit /b %errorlevel%
