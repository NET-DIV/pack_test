from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Union, List
from string import digits, ascii_letters


# --- Token Types ---
class TOKEN_TYPE(StrEnum):
    INT = auto()
    IDENT = auto()
    ASSIGN = auto()
    PRINT = auto()
    SET = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    STRING = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    FUNC = auto()
    RETURN = auto()
    COMMA = auto()
    LBRACE = auto()
    RBRACE = auto()


# --- Token ---
@dataclass
class Token:
    type: TOKEN_TYPE
    value: Any = None


# --- Tokenizer ---
class TOKENYZER:
    keywords = {
        "set": TOKEN_TYPE.SET,
        "print": TOKEN_TYPE.PRINT,
        "if": TOKEN_TYPE.IF,
        "else": TOKEN_TYPE.ELSE,
        "while": TOKEN_TYPE.WHILE,
        "func": TOKEN_TYPE.FUNC,
        "return": TOKEN_TYPE.RETURN,
    }

    def __init__(self, code: str, ptr: int = 0):
        self.code = code
        self.ptr = ptr

    def next_token(self) -> Token:
        while self.ptr < len(self.code) and self.code[self.ptr].isspace():
            self.ptr += 1

        if self.ptr >= len(self.code):
            return Token(TOKEN_TYPE.EOF)

        char = self.code[self.ptr]

        # Numbers
        if char in digits:
            start = self.ptr
            while self.ptr < len(self.code) and self.code[self.ptr] in digits:
                self.ptr += 1
            return Token(TOKEN_TYPE.INT, int(self.code[start:self.ptr]))

        # Identifiers or keywords
        if char in ascii_letters or char == '_':
            start = self.ptr
            while self.ptr < len(self.code) and (self.code[self.ptr] in ascii_letters or self.code[self.ptr] in digits or self.code[self.ptr] == '_'):
                self.ptr += 1
            word = self.code[start:self.ptr]
            if word in self.keywords:
                return Token(self.keywords[word])
            return Token(TOKEN_TYPE.IDENT, word)

        # Operators and punctuation
        # Two-char ops first
        if self.code.startswith("==", self.ptr):
            self.ptr += 2
            return Token(TOKEN_TYPE.EQ)
        if self.code.startswith("!=", self.ptr):
            self.ptr += 2
            return Token(TOKEN_TYPE.NE)
        if self.code.startswith("<=", self.ptr):
            self.ptr += 2
            return Token(TOKEN_TYPE.LE)
        if self.code.startswith(">=", self.ptr):
            self.ptr += 2
            return Token(TOKEN_TYPE.GE)

        # One-char ops
        if char == '+':
            self.ptr += 1
            return Token(TOKEN_TYPE.PLUS)
        if char == '-':
            self.ptr += 1
            return Token(TOKEN_TYPE.MINUS)
        if char == '*':
            self.ptr += 1
            return Token(TOKEN_TYPE.MUL)
        if char == '/':
            self.ptr += 1
            return Token(TOKEN_TYPE.DIV)
        if char == '=':
            self.ptr += 1
            return Token(TOKEN_TYPE.ASSIGN)
        if char == '(':
            self.ptr += 1
            return Token(TOKEN_TYPE.LPAREN)
        if char == ')':
            self.ptr += 1
            return Token(TOKEN_TYPE.RPAREN)
        if char == '{':
            self.ptr += 1
            return Token(TOKEN_TYPE.LBRACE)
        if char == '}':
            self.ptr += 1
            return Token(TOKEN_TYPE.RBRACE)
        if char == ',':
            self.ptr += 1
            return Token(TOKEN_TYPE.COMMA)
        if char == '"':
        	self.ptr += 1
        	start = self.ptr
        	while self.ptr < len(self.code) and self.code[self.ptr] != '"':
        		self.ptr += 1
        		if self.ptr >= len(self.code):
        			raise RuntimeError("Unterminated string literal")
        			string_value = self.code[start:self.ptr]
        			self.ptr += 1  # Skip closing "
        			return Token(TOKEN_TYPE.STRING, string_value)

        raise RuntimeError(f"Invalid character {char}")


# --- AST Nodes ---
@dataclass
class Number:
    value: int

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    left: Any
    op: TOKEN_TYPE
    right: Any

@dataclass
class Assign:
    name: str
    value: Any

@dataclass
class Print:
    value: Any

@dataclass
class If:
    condition: Any
    then_branch: Any
    else_branch: Any = None

@dataclass
class While:
    condition: Any
    body: Any

@dataclass
class Block:
    statements: List[Any]

@dataclass
class FuncDef:
    name: str
    params: List[str]
    body: Any

@dataclass
class FuncCall:
    name: str
    args: List[Any]

@dataclass
class String:
    value: str

@dataclass
class Return:
    value: Any


# --- Parser ---
class Parser:
    def __init__(self, tokenizer: TOKENYZER):
        self.tokenizer = tokenizer
        self.current_token = self.tokenizer.next_token()

    def eat(self, token_type: TOKEN_TYPE):
        if self.current_token.type == token_type:
            self.current_token = self.tokenizer.next_token()
        else:
            raise RuntimeError(f"Expected {token_type}, got {self.current_token.type}")

    def factor(self) -> Any:
        token = self.current_token
        if token.type == TOKEN_TYPE.INT:
            self.eat(TOKEN_TYPE.INT)
            return Number(token.value)
        elif token.type == TOKEN_TYPE.STRING:
        	self.eat(TOKEN_TYPE.STRING)
        	return String(token.value)
        elif token.type == TOKEN_TYPE.IDENT:
            self.eat(TOKEN_TYPE.IDENT)
            
            if self.current_token.type == TOKEN_TYPE.LPAREN:
                # Function call
                self.eat(TOKEN_TYPE.LPAREN)
                args = []
                if self.current_token.type != TOKEN_TYPE.RPAREN:
                    args.append(self.expr())
                    while self.current_token.type == TOKEN_TYPE.COMMA:
                        self.eat(TOKEN_TYPE.COMMA)
                        args.append(self.expr())
                self.eat(TOKEN_TYPE.RPAREN)
                return FuncCall(token.value, args)
            else:
                return Var(token.value)
        elif token.type == TOKEN_TYPE.LPAREN:
            self.eat(TOKEN_TYPE.LPAREN)
            node = self.expr()
            self.eat(TOKEN_TYPE.RPAREN)
            return node
        else:
            raise RuntimeError(f"Unexpected token in factor: {token}")

    def term(self) -> Any:
        node = self.factor()
        while self.current_token.type in (TOKEN_TYPE.MUL, TOKEN_TYPE.DIV):
            op = self.current_token.type
            self.eat(op)
            node = BinOp(left=node, op=op, right=self.factor())
        return node

    def expr(self) -> Any:
        node = self.term()
        while self.current_token.type in (TOKEN_TYPE.PLUS, TOKEN_TYPE.MINUS):
            op = self.current_token.type
            self.eat(op)
            node = BinOp(left=node, op=op, right=self.term())
        return node

    def comparison(self) -> Any:
        node = self.expr()
        if self.current_token.type in (TOKEN_TYPE.EQ, TOKEN_TYPE.NE, TOKEN_TYPE.LT, TOKEN_TYPE.GT, TOKEN_TYPE.LE, TOKEN_TYPE.GE):
            op = self.current_token.type
            self.eat(op)
            right = self.expr()
            node = BinOp(left=node, op=op, right=right)
        return node

    def statement(self) -> Any:
        if self.current_token.type == TOKEN_TYPE.SET:
            self.eat(TOKEN_TYPE.SET)
            if self.current_token.type != TOKEN_TYPE.IDENT:
                raise RuntimeError("Expected identifier after 'set'")
            var_name = self.current_token.value
            self.eat(TOKEN_TYPE.IDENT)
            self.eat(TOKEN_TYPE.ASSIGN)
            expr_value = self.comparison()
            return Assign(name=var_name, value=expr_value)

        elif self.current_token.type == TOKEN_TYPE.PRINT:
            self.eat(TOKEN_TYPE.PRINT)
            value = self.comparison()
            return Print(value=value)

        elif self.current_token.type == TOKEN_TYPE.IF:
            self.eat(TOKEN_TYPE.IF)
            self.eat(TOKEN_TYPE.LPAREN)
            condition = self.comparison()
            self.eat(TOKEN_TYPE.RPAREN)
            then_branch = self.block()
            else_branch = None
            if self.current_token.type == TOKEN_TYPE.ELSE:
                self.eat(TOKEN_TYPE.ELSE)
                else_branch = self.block()
            return If(condition, then_branch, else_branch)

        elif self.current_token.type == TOKEN_TYPE.WHILE:
            self.eat(TOKEN_TYPE.WHILE)
            self.eat(TOKEN_TYPE.LPAREN)
            condition = self.comparison()
            self.eat(TOKEN_TYPE.RPAREN)
            body = self.block()
            return While(condition, body)

        elif self.current_token.type == TOKEN_TYPE.FUNC:
            self.eat(TOKEN_TYPE.FUNC)
            if self.current_token.type != TOKEN_TYPE.IDENT:
                raise RuntimeError("Expected function name")
            name = self.current_token.value
            self.eat(TOKEN_TYPE.IDENT)
            self.eat(TOKEN_TYPE.LPAREN)
            params = []
            if self.current_token.type != TOKEN_TYPE.RPAREN:
                if self.current_token.type != TOKEN_TYPE.IDENT:
                    raise RuntimeError("Expected parameter name")
                params.append(self.current_token.value)
                self.eat(TOKEN_TYPE.IDENT)
                while self.current_token.type == TOKEN_TYPE.COMMA:
                    self.eat(TOKEN_TYPE.COMMA)
                    if self.current_token.type != TOKEN_TYPE.IDENT:
                        raise RuntimeError("Expected parameter name")
                    params.append(self.current_token.value)
                    self.eat(TOKEN_TYPE.IDENT)
            self.eat(TOKEN_TYPE.RPAREN)
            body = self.block()
            return FuncDef(name, params, body)

        elif self.current_token.type == TOKEN_TYPE.RETURN:
            self.eat(TOKEN_TYPE.RETURN)
            value = self.comparison()
            return Return(value)

        else:
            # Expression statement (function call or expression)
            node = self.comparison()
            return node

    def block(self) -> Block:
        self.eat(TOKEN_TYPE.LBRACE)
        statements = []
        while self.current_token.type != TOKEN_TYPE.RBRACE:
            statements.append(self.statement())
        self.eat(TOKEN_TYPE.RBRACE)
        return Block(statements)

    def parse(self) -> List[Any]:
        statements = []
        while self.current_token.type != TOKEN_TYPE.EOF:
            statements.append(self.statement())
        return statements
        
        
class Interpreter:
    def __init__(self):
        # Global scope and function definitions
        self.global_vars = {}
        self.functions = {}
        self.call_stack = []

    def eval(self, node, local_vars=None):
        if local_vars is None:
            local_vars = {}

        # Numbers
        if isinstance(node, Number):
            return node.value

        # Variables
        elif isinstance(node, Var):
            if node.name in local_vars:
                return local_vars[node.name]
            elif node.name in self.global_vars:
                return self.global_vars[node.name]
            else:
                raise RuntimeError(f"Undefined variable '{node.name}'")

        # Binary operations
        elif isinstance(node, BinOp):
            left = self.eval(node.left, local_vars)
            right = self.eval(node.right, local_vars)
            if node.op == TOKEN_TYPE.PLUS:
                return left + right
            elif node.op == TOKEN_TYPE.MINUS:
                return left - right
            elif node.op == TOKEN_TYPE.MUL:
                return left * right
            elif node.op == TOKEN_TYPE.DIV:
                return left // right  # integer division
            elif node.op == TOKEN_TYPE.EQ:
                return int(left == right)
            elif node.op == TOKEN_TYPE.NE:
                return int(left != right)
            elif node.op == TOKEN_TYPE.LT:
                return int(left < right)
            elif node.op == TOKEN_TYPE.GT:
                return int(left > right)
            elif node.op == TOKEN_TYPE.LE:
                return int(left <= right)
            elif node.op == TOKEN_TYPE.GE:
                return int(left >= right)
            else:
                raise RuntimeError(f"Unknown binary operator {node.op}")

        # Assignment
        elif isinstance(node, Assign):
            val = self.eval(node.value, local_vars)
            if node.name in local_vars:
            	local_vars[node.name] = val
            else:
            	self.global_vars[node.name] = val
            return val

        # Print statement
        elif isinstance(node, Print):
            val = self.eval(node.value, local_vars)
            print(val)
            return val

        # If statement
        elif isinstance(node, If):
            cond = self.eval(node.condition, local_vars)
            if cond:
                return self.eval_block(node.then_branch, local_vars)
            elif node.else_branch:
                return self.eval_block(node.else_branch, local_vars)

        # While loop
        elif isinstance(node, While):
            while self.eval(node.condition, local_vars):
                res = self.eval_block(node.body, local_vars)
                # Allow return from inside while
                if isinstance(res, Return):
                    return res
            return None

        # Block of statements
        elif isinstance(node, Block):
            return self.eval_block(node, local_vars)

        # Function definition
        elif isinstance(node, FuncDef):
            self.functions[node.name] = node
            return None

        # Function call
        elif isinstance(node, FuncCall):
            if node.name not in self.functions:
                raise RuntimeError(f"Undefined function '{node.name}'")
            func_def = self.functions[node.name]
            if len(node.args) != len(func_def.params):
                raise RuntimeError(f"Function '{node.name}' expects {len(func_def.params)} arguments but got {len(node.args)}")
            new_local_vars = {}
            for param, arg_node in zip(func_def.params, node.args):
                new_local_vars[param] = self.eval(arg_node, local_vars)
            self.call_stack.append(new_local_vars)
            ret = self.eval_block(func_def.body, new_local_vars)
            self.call_stack.pop()
            if isinstance(ret, Return):
                return self.eval(ret.value, new_local_vars)
            return None

        # Return statement
        elif isinstance(node, Return):
            return node

        else:
            raise RuntimeError(f"Unknown node type {type(node)}")

    def eval_block(self, block_node, local_vars):
        for stmt in block_node.statements:
            result = self.eval(stmt, local_vars)
            if isinstance(result, Return):
                return result
        return None
        
        
code = '''
set x = 5
print(x)
'''

tokenizer = TOKENYZER(code)
parser = Parser(tokenizer)
ast = parser.parse()

interpreter = Interpreter()
for statement in ast:
    interpreter.eval(statement)