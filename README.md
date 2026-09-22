# MLMP_project

Moder machine learning in practice: group project repo.

## Getting Started

### 1. Clone the Repository

To get a local copy of this public repository, open your terminal and run the following commands:

```bash
git clone https://github.com/LukaAgostini/MLMP_project.git
cd MLMP_project
```

### 2. Environment Setup

This project uses `uv` for fast environment and dependency management.

First, create and activate a virtual environment:

```bash
# Create the virtual environment
uv venv

# Activate the virtual environment (macOS/Linux)
source .venv/bin/activate

# Activate the virtual environment (Windows)
.venv\Scripts\activate
```

To install the existing project dependencies specified in `pyproject.toml`, run:

```bash
uv sync
```

To install new libraries and add them to the project's dependencies, use the following command:

```bash
uv add <package_name>
```

## Contributing

To ensure a smooth collaboration, especially when modifying code written by other contributors, please avoid pushing directly to the `main` branch. Instead, follow this workflow:

1. **Create a new branch** for your feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, commit, and push them to your branch.

3. **Open a Pull Request (PR)** against the `main` branch.

4. Wait for the PR to be reviewed and approved before merging your changes into the main codebase.