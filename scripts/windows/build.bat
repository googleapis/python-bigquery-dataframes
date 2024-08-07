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


echo "Listing available Python versions'
py -0

py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install --upgrade pip setuptools wheel

echo "Building Wheel"
py -3.10 -m pip wheel . --wheel-dir wheels/ || exit /b

echo "Built wheel, now running tests."
call %~dp0/test.bat 3.10 || exit /b

echo "Windows build has completed successfully"
