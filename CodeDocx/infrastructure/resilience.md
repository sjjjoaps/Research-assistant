# resilience.py

## 所在层次
基础设施层 `src/infrastructure/`

## 主体功能
重试与弹性工具模块，提供 `retry`（同步）、`async_retry`（异步）装饰器和 `CircuitBreaker` 熔断器。

## 关键类与方法

| 类/函数 | 作用 |
|---|---|
| `retry(max_retries, exceptions, delay)` | 同步重试装饰器，指定异常类型和延迟策略 |
| `async_retry(max_retries, exceptions, delay)` | 异步重试装饰器，适用于 `async def` 函数 |
| `CircuitBreaker` | 熔断器：连续失败超阈值时开路，冷却后半开探测 |

## 调用关系
- **被调用方**：LLM 调用、Neo4j 查询等可能短暂失败的外部 I/O 操作
- **依赖方**：无外部依赖（标准库 `time`、`asyncio`、`functools`）

## 注意事项
- `CircuitBreaker` 适用于外部服务频繁失败时保护主流程，避免级联故障
- 重试间隔可配置为指数退避
