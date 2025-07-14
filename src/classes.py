import ssl
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Any
from enum import Enum
import os
from pathlib import Path
import yaml
import json

@dataclass
class MessageEvent:
    namespace: str
    agent_name: str
    sent_at: str
    message: dict[str, Any]
    config: dict[str, Any]
    
    def __init__(
        self,
        rawEvent: dict[str, Any]
    ):
        # Mapping from typescript object to python object
        message = rawEvent["message"]
        self.namespace = rawEvent["namespace"]
        self.agent_name = rawEvent["agentName"]
        self.sent_at = rawEvent["sentAt"]
        self.message = {
            "id": message["id"],
            "thread": message["thread"],
            "content": message["content"],
            "role": message["role"]
        }
        self.config = rawEvent["config"]

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class SocketConfig:
    """Socket.IO configuration với validation"""
    url: str
    reconnection: bool = True
    reconnection_attempts: int = 10
    reconnection_delay: int = 1
    reconnection_delay_max: int = 5
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'SocketConfig':
        """Load socket config from environment"""
        return cls(
            url=os.getenv('JARVIS_KIT_RUNTIME', 'ws://localhost:3000'),
            reconnection=os.getenv('SOCKET_RECONNECTION', 'true').lower() == 'true',
            reconnection_attempts=int(os.getenv('SOCKET_RECONNECTION_ATTEMPTS', '10')),
            reconnection_delay=int(os.getenv('SOCKET_RECONNECTION_DELAY', '1')),
            reconnection_delay_max=int(os.getenv('SOCKET_RECONNECTION_DELAY_MAX', '5')),
            timeout=int(os.getenv('SOCKET_TIMEOUT', '30'))
        )
    
    def validate(self) -> None:
        """Validate configuration"""
        if not self.url:
            raise ValueError("Socket URL cannot be empty")
        if self.reconnection_attempts < 1:
            raise ValueError("Reconnection attempts must be >= 1")
        if self.timeout < 1:
            raise ValueError("Timeout must be >= 1")

@dataclass
class RabbitMQConfig:
    """RabbitMQ configuration với SSL support"""
    url: str
    ssl_context: ssl.SSLContext | None = None
    connection_timeout: int = 30
    heartbeat: int = 600
    prefetch_count: int = 10
    
    @classmethod
    def from_env(cls) -> 'RabbitMQConfig':
        """Load RabbitMQ config from environment"""
        ssl_context = None
        if os.getenv('RABBITMQ_SSL_ENABLED', 'false').lower() == 'true':
            ssl_context = ssl.create_default_context()
            if cert_file := os.getenv('RABBITMQ_SSL_CERT'):
                ssl_context.load_cert_chain(cert_file)
        
        return cls(
            url=os.getenv('RABBITMQ_URL', 'amqp://localhost'),
            ssl_context=ssl_context,
            connection_timeout=int(os.getenv('RABBITMQ_CONNECTION_TIMEOUT', '30')),
            heartbeat=int(os.getenv('RABBITMQ_HEARTBEAT', '600')),
            prefetch_count=int(os.getenv('RABBITMQ_PREFETCH_COUNT', '10'))
        )
    
    def validate(self) -> None:
        """Validate RabbitMQ configuration"""
        if not self.url:
            raise ValueError("RabbitMQ URL cannot be empty")
        if self.connection_timeout < 1:
            raise ValueError("Connection timeout must be >= 1")

@dataclass
class RuntimeMessage(TypedDict):
    id: str
    thread: str
    content: str
    role: Literal["user", "agent"]
    toolCallId: str
    toolName: str
    toolInput: dict[str, Any]
    toolResults: dict[str, Any]
    metadata: dict[str, Any]
    createdAt: str
    updatedAt: str

@dataclass
class ClientResponseData(TypedDict):
    namespace: str
    agent_name: str
    tool_call_id: str
    response: Any

@dataclass
class RuntimeConfig:
    """Centralized runtime configuration với environment support"""
    namespace: str
    namespace_api_key: str
    runtime_endpoint: str
    socket_config: SocketConfig
    rabbitmq_config: RabbitMQConfig
    environment: Environment = Environment.DEVELOPMENT
    max_concurrent_workers: int = 2
    timeout: int = 30
    debug: bool = False
    redis_url: str | None = None
    metrics_enabled: bool = True
    health_check_interval: int = 30
    
    @classmethod
    def from_env(cls) -> 'RuntimeConfig':
        """Load config from environment variables"""
        config = cls(
            namespace=os.getenv('JARVIS_KIT_NAMESPACE', ''),
            namespace_api_key=os.getenv('JARVIS_KIT_NAMESPACE_SECRET', ''),
            runtime_endpoint=os.getenv('JARVIS_KIT_RUNTIME', ''),
            socket_config=SocketConfig.from_env(),
            rabbitmq_config=RabbitMQConfig.from_env(),
            environment=Environment(os.getenv('ENVIRONMENT', 'development')),
            max_concurrent_workers=int(os.getenv('MAX_CONCURRENT_WORKERS', '2')),
            timeout=int(os.getenv('CONNECTION_TIMEOUT', '30')),
            debug=os.getenv('DEBUG', 'false').lower() == 'true',
            redis_url=os.getenv('REDIS_URL'),
            metrics_enabled=os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
        )
        config.validate()
        return config
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'RuntimeConfig':
        """Load config from YAML/JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**data)
    
    def validate(self) -> None:
        """Validate all configuration"""
        if not self.namespace:
            raise ValueError("Namespace cannot be empty")
        if not self.namespace_api_key:
            raise ValueError("Namespace API key cannot be empty")
        if not self.runtime_endpoint:
            raise ValueError("Runtime endpoint cannot be empty")
        if self.max_concurrent_workers < 1:
            raise ValueError("Max concurrent workers must be >= 1")
        
        # Validate nested configs
        self.socket_config.validate()
        self.rabbitmq_config.validate()
        
        # Environment-specific validation
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("Debug mode should be disabled in production")
            if not self.redis_url:
                print("You are running in production mode without using RedisMessageStore. This is not recommended.")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'namespace': self.namespace,
            'environment': self.environment.value,
            'max_concurrent_workers': self.max_concurrent_workers,
            'timeout': self.timeout,
            'debug': self.debug,
            'metrics_enabled': self.metrics_enabled,
            'health_check_interval': self.health_check_interval
        }