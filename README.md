# Suanli Task Manager Main Conc

面向多项目的异步任务网关。

当前版本的核心特性：

- 统一外部入口，内部按 project 走独立 adapter
- 每个 project 独立排队、独立调度、独立空闲暂停
- 请求接入与运行时调度解耦，支持并发创建任务和查询任务
- 上传文件先落盘再入队，不再把原始字节长期保存在内存里
- 终态任务默认保留 1 小时，后台定期清理过期记录和残留目录

## 运行

```bash
uv run main.py
```

或：

```bash
uv run python main.py --config config.yaml
```

## 测试

```bash
uv run pytest tests -q
```

## 配置说明

`config.yaml` 现在包含两类配置：

- `server`
  - `temp_root`：任务临时目录根路径
  - `cleanup_interval_sec`：后台清理周期
  - `default_task_ttl_sec`：终态任务默认保留时长
- `projects.<name>`
  - `adapter`：项目适配器名称
  - `queue_limit`：项目独立排队上限
  - `health_path/ready_path/live_path`：runtime 探针路径
  - Task Manager 在 runtime 恢复并拿到 `service_url` 后，只会等待该项目配置的 `ready_path` 成功，再开始派发任务
  - 其余 `task_id/service_port/token/...` 仍表示平台运行配置
