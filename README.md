# deep_rl_101

Deep Rl 101 project

## Features
- **Dask** distributed computing
- **Loguru** structured logging
- **Ruff** linting and formatting
- **UV** package management
- **Click** CLI support

## Documentation

- [PPO 模型架构](docs/model-architecture.md) - PPO 模型与环境空间关联机制详解

## Installation

1. **Clone the template:**
   ```bash
   copier copy <template-source> deep_rl_101
   ```

2. **Install dependencies:**
   ```bash
   cd deep_rl_101
   uv sync
   ```

## Usage

### Development

```bash
# Start the development server
make dev
```


## Project Structure

```
deep_rl_101/
├── config/
│   └── config.yaml          # Configuration file
├── docs/
│   └── model-architecture.md # PPO 模型架构文档
├── scripts/
│   └── inspect_ppo_model.py # PPO 模型检查脚本
├── src/
│   ├── __init__.py
│   ├── app.py              # Main application logic
│   ├── main.py             # Entry point
│   └── utils/              # Utility modules
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── custom_logging.py
│       ├── dask.py         # Dask utilities
│       └── utils.py
├── Makefile                # Common commands
├── pyproject.toml          # Project configuration
├── copier.yml              # Template configuration
└── README.md               # This file
```

## Configuration

The configuration is managed through `config/config.yaml`:

- **HTTP Server**: Port 13000
- **Dask Cluster**: Scheduler port 8786, Dashboard port 8787
- **Logging**: Level INFO, rotation 1 days, retention 5 days

## Development

### Code Quality

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .
```

### Testing
To enable testing, set `include_tests: true` when generating the project.

## Template Variables

This template uses the following variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `project_name` | Project name | `py-project-template` |
| `project_description` | Project description | `Add your description here` |
| `author_name` | Author name | `""` |
| `author_email` | Author email | `""` |
| `python_version` | Python version | `3.12` |
| `use_dask` | Enable Dask | `true` |
| `use_click` | Enable Click CLI | `true` |
| `use_pydub` | Enable pydub | `true` |
| `http_port` | HTTP server port | `13000` |
| `dask_scheduler_port` | Dask scheduler port | `8786` |
| `dask_dashboard_port` | Dask dashboard port | `8787` |
| `log_level` | Logging level | `INFO` |
| `log_rotation` | Log rotation period | `1 days` |
| `log_retention` | Log retention period | `5 days` |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under MIT License.

## Acknowledgments

- [Copier](https://copier.readthedocs.io/) for project templating
- [Dask](https://dask.org/) for distributed computing
- [UV](https://docs.astral.sh/uv/) for package management
- [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
