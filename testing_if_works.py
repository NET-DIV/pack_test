import os
try:
	import telebot
except:
	os.system("pip install telebot")
try:
	import torch
except:
	os.system("pip install torch")

import torch.nn as nn
import torch
import torch.nn.functional as F
try:
	import time
except:
	os.system("pip install time")
try:
	from collections import Counter
except:
	os.system("pip install collections")

from typing import Any
from enum import StrEnum, auto
from dataclasses import dataclass
from string import digits, ascii_letters
bot_token = "7857179811:AAE1JtxjzyXrhdjoA-xL3B6bkD5DMKJK7-0"
bot = telebot.TeleBot(token=bot_token)
def uid(message):
    return message.chat.id
class TokenType(StrEnum):
    INT = auto()
    STRING = auto()
    COLOM = auto()
    SEM = auto()
    PLUS = auto()
    RETURN = auto()
    WHILE = auto()
    FOR = auto()
    ELSE = auto()
    IF = auto()
    PRINT = auto()
    FLOAT = auto()
    DOUBLE = auto()
    VOID = auto()
    FUNCTION = auto()
    POINTER = auto()
    DECIMAL = auto()
    CHAR = auto()
    MINUS = auto()
    SET = auto()
    LET = auto()
    VAR = auto()
    CONST = auto()
    UNDEF = auto()
    PARAM = auto()
    LBRAC = auto()
    RBRAC = auto()
    LWING = auto()
    RWING = auto()
    LCUBE = auto()
    STRING2 = auto()
    MALLOC = auto()
    RCUBE = auto()
    INCLUDE = auto()
    DEFINE = auto()
    CLASS = auto()
    OBJECT = auto()
    STATICMETHOD = auto()
    DATACLASS = auto()
    INIT = auto()
    POINT = auto()
    TRY = auto()
    EXCEPT = auto()
    SWITCH = auto()
    EOF = auto()
    TIME = auto()
    DEVIDE = auto()
    ELIF = auto()
    COUT = auto()
    ENDIF = auto()
    CIN = auto()
    SYSTEM = auto()
    ENTER = auto()
    OS = auto()
    WB = auto()
    EXIT = auto()
    IDENT = auto()
    HEXPOSITION = auto()
    BTHEN = auto()
    EQU = auto()
    EXLAMANATION = auto()
    IFNDEF = auto()
    ARGS = auto()
    INTERNAL = auto()
    SIZEOF= auto()
    NEQ = auto()
    STATIC = auto()
    AT_TAG = auto()
    LEQ = auto()
    PERCENT = auto()
    EQ = auto()
    Ampersand = auto()
    HASHTAG = auto()
    EQUAL = auto()
    IFDEF = auto()
    EXTERNAL = auto()
    GEQ = auto()
    PUBLIC = auto()
    GOTO = auto()
    PRIVATE = auto()
    STRUCT = auto()
    CODE = auto()
    DO = auto()
    SINGLE_COTATION = auto()
    BOOLEAN = auto()
    OCTAL = auto()
    BINARY = auto()
    DOUBLE_COTATION = auto()
    DEFINED = auto()
    TYPE_DEF = auto()
    LTHEN = auto()

@dataclass
class TOKEN:
    type: TokenType
    value: Any = None

class Tokenizer:
    def __init__(self, code: str, ptr=0):
        self.code = code
        self.ptr = ptr
        self.token_to_id = {token: idx for idx, token in enumerate(TokenType)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.keywords = {
            "int": TokenType.INT,
            "string":TokenType.STRING,
            "String": TokenType.STRING,
            ",": TokenType.COLOM,
            ";": TokenType.SEM,
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.TIME,
            "/": TokenType.DEVIDE,
            "return": TokenType.RETURN,
            "void": TokenType.VOID,
            "float": TokenType.FLOAT,
            "internal":TokenType.INTERNAL,
            "'":TokenType.SINGLE_COTATION,
            '#"':TokenType.DOUBLE_COTATION,
            "external":TokenType.EXTERNAL,
            "double": TokenType.DOUBLE,
            "var": TokenType.VAR,
            "#defined":TokenType.DEFINED,
            "let": TokenType.LET,
            "const": TokenType.CONST,
            "%d": TokenType.DECIMAL,
            "undef":TokenType.UNDEF,
            "endif":TokenType.ENDIF,
            "sys":TokenType.SYSTEM,
            "system":TokenType.SYSTEM,
            "System":TokenType.SYSTEM,
            "enter":TokenType.ENTER,
            "Enter":TokenType.ENTER,
            "__enter__":TokenType.ENTER,
            "os":TokenType.OS,
            "__os__":TokenType.OS,
            "['os']":TokenType.OS,
            "wb":TokenType.WB,
            "__wb__":TokenType.WB,
            "exit":TokenType.EXIT,
            "Exit":TokenType.EXIT,
            "__exit__":TokenType.EXIT,
            "code":TokenType.CODE,
            "Code":TokenType.CODE,
            "ifndef":TokenType.IFNDEF,
            "#ifndef":TokenType.IFNDEF,
            "!":TokenType.EXLAMANATION,
            "bool":TokenType.BOOLEAN,
            "boolean":TokenType.BOOLEAN,
            "typedef":TokenType.TYPE_DEF,
            "#typedef":TokenType.TYPE_DEF,
            "sizeof":TokenType.SIZEOF,
            "args":TokenType.ARGS,
            "args[]":TokenType.ARGS,
            "static":TokenType.STATIC,
            "public":TokenType.PUBLIC,
            "do":TokenType.DO,
            "goto":TokenType.GOTO,
            "private":TokenType.PRIVATE,
            "ifdef":TokenType.IFDEF,
            "#ifdef":TokenType.IFDEF,
            "%s": TokenType.STRING2,
            "function": TokenType.FUNCTION,
            "(": TokenType.RBRAC,
            ")": TokenType.LBRAC,
            "}": TokenType.LWING,
            "{": TokenType.RWING,
            "]": TokenType.LCUBE,
            "[": TokenType.RCUBE,
            "struct":TokenType.STRUCT,
            "#include": TokenType.INCLUDE,
            ">": TokenType.BTHEN,
            "<": TokenType.LTHEN,
            "while":TokenType.WHILE,
            "#while": TokenType.WHILE,
            "if":TokenType.IF,
            "#if": TokenType.IF,
            "else":TokenType.ELSE,
            "#else": TokenType.ELSE,
            "%":TokenType.PERCENT,
            "elif":TokenType.ELIF,
            "#elif": TokenType.ELIF,
            "#define":TokenType.DEFINE,
            "define": TokenType.DEFINE,
            "class": TokenType.CLASS,
            "object": TokenType.OBJECT,
            "staticmethod": TokenType.STATICMETHOD,
            "&":TokenType.Ampersand,
            "dataclass": TokenType.DATACLASS,
            "__init__": TokenType.INIT,
            ".": TokenType.POINT,
            "malloc": TokenType.MALLOC,
            "print": TokenType.PRINT,
            "switch":TokenType.SWITCH,
            "#":TokenType.HASHTAG,
            "@":TokenType.AT_TAG,
            "try":TokenType.TRY,
            "#try":TokenType.TRY,
            "except":TokenType.EXCEPT,
            "#except":TokenType.EXCEPT,
            "cout":TokenType.COUT,
            "std::cout": TokenType.COUT,
            "cin":TokenType.CIN,
            "std::cin": TokenType.CIN,
            "0x": TokenType.HEXPOSITION,
            "0o":TokenType.OCTAL,
            "0b":TokenType.BINARY,
            "=": TokenType.EQU,
            "==": TokenType.EQUAL,
            "!=": TokenType.NEQ,
            "<=": TokenType.LEQ,
            ">=": TokenType.GEQ
        }

    def get_token(self) -> TOKEN:
        while self.ptr < len(self.code) and self.code[self.ptr].isspace():
            self.ptr += 1

        if self.ptr >= len(self.code):
            return TOKEN(TokenType.EOF)

        char = self.code[self.ptr]
        if char == "0":
        	start = self.ptr
        	self.ptr += 1
        	if self.ptr < len(self.code):
        		next_char = self.code[self.ptr].lower()
        		if next_char == "x":
        		  self.ptr += 1
        		  while self.ptr < len(self.code) and self.code[self.ptr].lower() in "0123456789abcdef":
        		  	self.ptr += 1
        		  return TOKEN(TokenType.HEXPOSITION, self.code[start:self.ptr])
        		elif next_char == "o":
        		  self.ptr += 1
        		  while self.ptr < len(self.code) and self.code[self.ptr] in "01234567":
        		  	self.ptr += 1
        		  return TOKEN(TokenType.OCTAL, self.code[start:self.ptr])
        		elif next_char == "b":
        		  self.ptr += 1
        		  while self.ptr < len(self.code) and self.code[self.ptr] in "01":
        		  	self.ptr += 1
        		  return TOKEN(TokenType.BINARY, self.code[start:self.ptr])
        		elif self.code[self.ptr] in digits:
        		      while self.ptr < len(self.code) and self.code[self.ptr] in digits:
        		      	self.ptr += 1
        		      return TOKEN(TokenType.INT, self.code[start:self.ptr])
        	return TOKEN(TokenType.INT, "0")
        if char in digits:
        	start = self.ptr
        	while self.ptr < len(self.code) and self.code[self.ptr] in digits:
        		self.ptr += 1
        	return TOKEN(TokenType.INT, self.code[start:self.ptr])

            
        if char in ascii_letters or char == "_":
        	start = self.ptr
        	while self.ptr < len(self.code) and (
        	self.code[self.ptr] in ascii_letters or self.code[self.ptr] in digits or self.code[self.ptr] == "_"):
        		self.ptr += 1
        	word = self.code[start:self.ptr]
        	lowered = word.lower()
        	if lowered in self.keywords:
        		return TOKEN(self.keywords[lowered], word)
        	return TOKEN(TokenType.IDENT, word)
        if self.code.startswith("==", self.ptr):
            self.ptr += 2
            return TOKEN(TokenType.EQUAL)
        if self.code.startswith("!=", self.ptr):
            self.ptr += 2
            return TOKEN(TokenType.NEQ)
        if self.code.startswith("<=", self.ptr):
            self.ptr += 2
            return TOKEN(TokenType.LEQ)
        if self.code.startswith(">=", self.ptr):
            self.ptr += 2
            return TOKEN(TokenType.GEQ)

        self.ptr += 1
        return TOKEN(self.keywords.get(char, TokenType.IDENT), char)
    def get_token_id(self, tokentype):
    	return self.token_to_id(tokentype - 1)
    def get_token_by_id(self, idx):
    	return self.id_to_token.get(idx, None)
    def token_(self):
    	return self.token_to_id


class EVOB:
    def __init__(self, tokenizer_instance, embed_dim=128):
        self.token_to_ids = tokenizer_instance.token_()  
        self.vocab_size = len(self.token_to_ids)
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

    def learning(self, ids):
        tokens_tensor = torch.tensor(ids, dtype=torch.long)
        embeddings = self.embedding(tokens_tensor)
        return embeddings  
    def ids_from_tokens(self, tokens):
        return [self.token_to_ids[tok.type] for tok in tokens]
		
class Trainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.embedding.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, input_ids, target_ids):
        self.model.embedding.train()
        self.optimizer.zero_grad()

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        outputs = self.model.embedding(input_tensor)
        logits = outputs @ self.model.embedding.weight.T 
        loss = self.criterion(logits, target_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.item()
class ContextGenerator:
    def __init__(self, window_size=2):
        self.window_size = window_size

    def generate(self, tokens):
        pairs = []
        for i, token in enumerate(tokens):
            for j in range(max(i - self.window_size, 0), min(i + self.window_size + 1, len(tokens))):
                if i != j:
                    pairs.append((tokens[i], tokens[j]))
        return pairs
    
    
class VectorAnalyzer:
    def __init__(self, embedding_layer, id_to_token, pit):
        self.embeddings = embedding_layer.weight.data  
        self.id_to_token = id_to_token
        self.pit = pit
    def cosine_similarity(self, vec1, vec2):
        dot = torch.dot(vec1, vec2)
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        return dot / (norm1 * norm2 + 1e-8) 
    def show_similar_tokens(self, token_id, top_k=5):
        target_vec = self.embeddings[token_id]
        similarities = []

        for idx, vec in enumerate(self.embeddings):
            if idx == token_id:
                continue 
            sim = self.cosine_similarity(target_vec, vec)
            similarities.append((idx, sim.item()))
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ids = similarities[:top_k]

        print(f"\nTop {top_k} similar tokens to: {self.id_to_token[token_id].name}")
        for idx, score in top_ids:
            token = self.id_to_token[idx]
            print(f"  {token.name} (score: {score:.4f})")
    def analyzer(self,):
    	for tokenn in self.id_to_token:
    		embeder = torch.binary_cross_entropy_with_logits[tokenn]
    		mapp = torch.tensor([embeder])
    		return mapp
    def reply_to_meaning(self,):
    	vec = self.embeddings[Tokenizer.token_]
    	simil = torch.are_deterministic_algorithms_enabled
    	vec[simil]
    	self.embeddings.append(vec)
    	return simil
    def reasoning_to_reply(self, reasons):
    	vectorizing = torch.conv_transpose3d()
    	simi = []
    	while vectorizing is True:
    		X = vectorizing.ectorAnalyzer[self.cosine_similarity]
    		Y = vectorizing.ectorAnalyzer[self.embeddings]
    		simi.append(X , Y, self.embeddings.items())
    		return simi
@bot.message_handler(content_types=['document'])
def handle_document(message):
    user_id = message.from_user.id
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    filename = f"{int(time.time())}_{message.document.file_name}"

    with open(filename, "wb") as f:
        f.write(downloaded_file)

    bot.reply_to(message, f"`{filename}` was saved successfully", parse_mode='Markdown')

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    tokenizer = Tokenizer(content)
    tokens = []
    raw_tokens = []

    while True:
        token = tokenizer.get_token()
        if token.type == TokenType.EOF:
            break
        tokens.append(f"{token.type.name}: {token.value}")
        raw_tokens.append(token)  
    token_message = "\n".join(tokens)
    if len(token_message) > 4000:
        token_message = token_message[:4000] + "\n... (truncated)"
    bot.send_message(user_id, f"```\n{token_message}\n```", parse_mode="Markdown")
    token_message = "\n".join(tokens)
    output_filename = f"tokens_{int(time.time())}.txt"
    token_ids = [tokenizer.token_to_id[token.type] for token in raw_tokens]
    evo = EVOB(tokenizer_instance=tokenizer)

  
    embeddings = evo.learning(token_ids)

 
    bot.send_message(user_id, f"Processed {len(token_ids)} tokens. Embeded: {embeddings.shape}")
    with open(output_filename, "w", encoding="utf-8") as f:
    	f.write(token_message + "\n\n")
    	f.write("Token IDs:\n" + ', '.join(map(str, token_ids)) + "\n\n")
    	f.write("Embeddings shape:\n" + str(embeddings.shape) + "\n")
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
    		with open(output_filename, "rb") as f:
    			bot.send_document(user_id, f)
    else:
    		bot.send_message(user_id, "Cannot write on an emty file [Bad request 400]")
 
bot.polling()