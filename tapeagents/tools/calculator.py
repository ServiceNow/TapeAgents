# derived from https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py

from __future__ import division

import json
import math
import operator
import re
from typing import Any, Literal

from pydantic import Field
from pyparsing import (
    CaselessLiteral,
    Combine,
    Forward,
    Group,
    Literal as ParsingLiteral,
    Optional,
    Word,
    ZeroOrMore,
    alphas,
    nums,
    oneOf,
)

from tapeagents.core import Action, Observation
from tapeagents.tools.base import Tool


def cmp(a, b):
    return (a > b) - (a < b)


class NumericStringParser(object):
    def pushFirst(self, strg, loc, toks):
        self.exprStack.append(toks[0])

    def pushUMinus(self, strg, loc, toks):
        if toks and toks[0] == "-":
            self.exprStack.append("unary -")

    def __init__(self):
        """
        expop   :: '^'
        multop  :: '*' | '/'
        addop   :: '+' | '-'
        integer :: ['+' | '-'] '0'..'9'+
        atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
        factor  :: atom [ expop factor ]*
        term    :: factor [ multop factor ]*
        expr    :: term [ addop term ]*
        """
        point = ParsingLiteral(".")
        e = CaselessLiteral("E")
        fnumber = Combine(
            Word("+-" + nums, nums) + Optional(point + Optional(Word(nums))) + Optional(e + Word("+-" + nums, nums))
        )
        ident = Word(alphas, alphas + nums + "_$")
        plus = ParsingLiteral("+")
        minus = ParsingLiteral("-")
        mult = ParsingLiteral("*")
        div = ParsingLiteral("/")
        lpar = ParsingLiteral("(").suppress()
        rpar = ParsingLiteral(")").suppress()
        addop = plus | minus
        multop = mult | div
        expop = ParsingLiteral("^")
        pi = CaselessLiteral("PI")
        expr = Forward()
        atom = (
            (Optional(oneOf("- +")) + (ident + lpar + expr + rpar | pi | e | fnumber).setParseAction(self.pushFirst))
            | Optional(oneOf("- +")) + Group(lpar + expr + rpar)
        ).setParseAction(self.pushUMinus)
        # by defining exponentiation as "atom [ ^ factor ]..." instead of
        # "atom [ ^ atom ]...", we get right-to-left exponents, instead of left-to-right
        # that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor << atom + ZeroOrMore((expop + factor).setParseAction(self.pushFirst))
        term = factor + ZeroOrMore((multop + factor).setParseAction(self.pushFirst))
        expr << term + ZeroOrMore((addop + term).setParseAction(self.pushFirst))
        # addop_term = ( addop + term ).setParseAction( self.pushFirst )
        # general_term = term + ZeroOrMore( addop_term ) | OneOrMore( addop_term)
        # expr <<  general_term
        self.bnf = expr
        # map operator symbols to corresponding arithmetic operations
        epsilon = 1e-12
        self.opn = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "^": operator.pow}
        self.fn = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "abs": abs,
            "trunc": lambda a: int(a),
            "round": round,
            "sgn": lambda a: abs(a) > epsilon and cmp(a, 0) or 0,
        }

    def evaluateStack(self, s):
        op = s.pop()
        if op == "unary -":
            return -self.evaluateStack(s)
        if op in "+-*/^":
            op2 = self.evaluateStack(s)
            op1 = self.evaluateStack(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op in self.fn:
            return self.fn[op](self.evaluateStack(s))
        elif op[0].isalpha():
            return 0
        else:
            return float(op)

    def eval(self, num_string, parseAll=True):
        self.exprStack = []
        self.bnf.parseString(num_string, parseAll)
        val = self.evaluateStack(self.exprStack[:])
        return val


def calculate(expr: str, values_dict: dict[str, Any]) -> str:
    pairs = values_dict.items()
    pairs = sorted(pairs, key=lambda x: len(x[0]), reverse=True)  # substitute longest vars first
    for k, v in pairs:
        if isinstance(v, str):
            expr = re.sub(k, "'{}'".format(v), expr)
        else:
            expr = re.sub(k, str(v), expr)
    try:
        result = NumericStringParser().eval(expr)
    except Exception:
        raise ValueError(f"Error evaluating expression: {expr}")
    try:
        str_result = json.dumps(result)
    except Exception:
        str_result = str(result)
    return str_result


class UseCalculatorAction(Action):
    """
    Action to use calculator to find the new fact. This python math expression uses only the fact names from the previous steps and constants. The expression should be a single line. You can use exp, cos, sin, tan, abs, trunc, sgn, round
    """

    kind: Literal["use_calculator_action"] = "use_calculator_action"
    expression: str = Field(description="math expression using previously known fact names and constants")
    fact_name: str = Field(
        description="fact name to save calculations result, should be unique, lowercase, snake_case, without spaces and special characters"
    )
    fact_unit: str = Field(description="expected unit of the fact value, if applicable, otherwise empty string")
    facts: dict | None = None


class CalculationResultObservation(Observation):
    kind: Literal["calculation_result_observation"] = "calculation_result_observation"
    name: str
    result: str


class Calculator(Tool):
    """
    Tool to evaluate math expressions
    """

    action: type[Action] = UseCalculatorAction
    observation: type[Observation] = CalculationResultObservation

    def execute_action(self, action: UseCalculatorAction) -> CalculationResultObservation:
        result = calculate(action.expression, action.facts or {})
        return CalculationResultObservation(name=action.fact_name, result=result)
