"""Контракты затратных запросов и сохраняемых конфигураций."""
from typing import Optional, Literal, Dict, Any, List, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def get(self,key,default=None):
        return getattr(self,key,default)


class BotBuildRequest(RequestModel):
    description: str = Field(min_length=1,max_length=10000)
    model: Optional[str] = Field(default=None,max_length=128)
    max_iterations: int = Field(default=3,ge=0,le=3,strict=True)


class BotChatRequest(RequestModel):
    spec: dict = Field(default_factory=dict)
    message: str = Field(min_length=1,max_length=100000)
    history: list[dict] = Field(default_factory=list,max_length=128)
    model: Optional[str] = Field(default=None,max_length=128)


class BotReviseRequest(RequestModel):
    spec: dict
    feedback: str = Field(min_length=1,max_length=10000)
    model: Optional[str] = Field(default=None,max_length=128)


class DatasetBuildRequest(RequestModel):
    dataset_id: Optional[str] = Field(default=None,pattern="^[a-f0-9]{32}$")
    dataset_path: Optional[str] = Field(default=None,max_length=4096)
    description: str = Field(default="",max_length=10000)
    target_column: Optional[str] = Field(default=None,max_length=200)


class RetrainRequest(RequestModel):
    dataset_path: str = Field(min_length=1,max_length=4096)
    target: Optional[str] = Field(default=None,max_length=200)
    target_column: Optional[str] = Field(default=None,max_length=200)
    model_type: Optional[str] = Field(default=None,max_length=128)
    new_data: bool = False
    only_if_weak: bool = False
    budget: Optional[Literal["fast","balanced","thorough"]] = None


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(min_length=1,max_length=200)
    description: str = Field(default="",max_length=10000)
    system_prompt: str = Field(max_length=100000)
    greeting: str = Field(default="Hi!",max_length=10000)
    temperature: float = Field(default=0.5,ge=0,le=2,allow_inf_nan=False)
    max_tokens: int = Field(default=1000,ge=1,le=4096,strict=True)
    tools: list[str] = Field(default_factory=list,max_length=64)
    price_per_call: float = Field(default=0,ge=0,le=1000000,allow_inf_nan=False)

    @field_validator("price_per_call")
    @classmethod
    def price(cls,value):
        from app.core.money import price_cents
        price_cents(value)
        return value


class MessageRequest(BaseModel):
    background: bool = False
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    message: str = Field(max_length=100000); conversation_id: Optional[str] = Field(default=None, max_length=200)
    projects: Optional[List[Dict]] = None
    attachments: Optional[List[str]] = None   # ids returned by POST /api/agent/upload


class PublishRequest(BaseModel):
    model_id: str; title: str; description: str
    task: str; model_type: str
    metrics: Dict[str, float] = {}; price_per_call: float = Field(default=0.01, ge=0, le=1000000, allow_inf_nan=False)

    @field_validator("price_per_call")
    @classmethod
    def valid_price(cls, value):
        from app.core.money import price_cents
        price_cents(value)
        return value


class RegisterRequest(BaseModel):
    email: str; password: str; name: str = ""
    # bot protection (app/core/antibot.py): hidden honeypot, ms the form was open, Cloudflare Turnstile token
    website: str = ""; form_elapsed_ms: Optional[float] = None; captcha_token: str = ""


class LoginRequest(BaseModel):
    email: str; password: str


class ChatCompletionMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str = Field(max_length=100000)


class ChatCompletionsRequest(BaseModel):
    """Body for POST /api/v1/chat/completions - the public developer API. Deliberately named
    and shaped like OpenAI's chat-completions request so existing tooling/snippets built for
    that shape need minimal changes to point at this platform instead."""
    model: Optional[str] = None
    messages: List[ChatCompletionMessage] = Field(min_length=1, max_length=128)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)


class UseModelRequest(BaseModel):
    features: List[Union[float, str]] = Field(min_length=1, max_length=10000)   # raw values; categorical models take strings


class CheckoutRequest(BaseModel):
    amount_usd: float


class SubscriptionCheckoutRequest(BaseModel):
    plan_id: str
