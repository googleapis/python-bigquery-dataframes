# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta


class MemberSelector(metaclass=ABCMeta):
    """
    Small helper class allowing class variables to relief the user from remembering strings or alike. See example below
    """
    # T_ for Type.... change on class level if necessary!
    _prefix: str = "T_"

    @classmethod
    def methods(cls, prefix: str|None=None) -> list:
        pref = prefix if prefix is not None else cls._prefix
        return [cls.__dict__[member] for member in cls.__dict__ if member.startswith(pref)]

    @classmethod
    def identifiers(cls, prefix: str|None=None) -> list:
        pref = prefix if prefix is not None else cls._prefix
        return [c for c in dir(cls) if c.startswith(pref)]
    
    @classmethod
    def methods_except(cls, exceptions: list[str], prefix: str|None=None) -> list:
        pref = prefix if prefix is not None else cls._prefix
        return [m for m in cls.methods() if m not in exceptions]
