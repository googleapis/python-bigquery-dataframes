#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import typing

import bigframes.core.transpiler.googlesql.expression as expr
import bigframes.core.transpiler.googlesql.from_ as from_
import bigframes.core.transpiler.googlesql.sql as sql

"""Python classes for GoogleSQL UNNEST operator, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#unnest_operator"""


@dataclasses.dataclass
class UnnestOperator(sql.SQLSyntax):
    """GoogleSQL unnest_operator syntax."""

    array: expr.ABCExpression
    as_alias: typing.Optional[expr.AsAlias] = None
    offset_alias: typing.Optional[expr.AsAlias] = None

    def sql(self) -> str:
        text = f"UNNEST({self.array.sql()})"
        if self.as_alias is not None:
            text = f"{text} {self.as_alias.sql()}"
        if self.offset_alias is not None:
            text = f"{text}\nWITH OFFSET {self.offset_alias.sql()}"
        return text
