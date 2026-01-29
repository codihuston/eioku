# Development Principles

## Core Principles

### Incremental Development
- Make the smallest possible changes that add value
- Each change should be independently testable and reviewable
- Prefer multiple small commits over large monolithic changes
- Build functionality piece by piece, validating at each step

### Commit Approval Workflow
- **ALWAYS** prompt for approval before committing any code changes
- Present a clear summary of what will be changed
- Wait for explicit user confirmation before proceeding
- Allow user to review and modify changes before commit

### Code Review Process
- **ALL** code changes must be reviewed using `principal-software-engineer.agent.md`
- Submit code for architectural and quality review before commit
- Address any feedback or recommendations from the review
- Only proceed with commit after review approval

### Change Management
- Show diffs for all file modifications
- Explain the reasoning behind each change
- Highlight any potential impacts or dependencies
- Provide rollback options when possible

### Iterative Implementation
- **Work iteratively with as many commits and tests as possible**
- **Each commit should be testable unless explicitly stated otherwise**
- **One schema per commit** - Database changes in small, focused commits
- **Test-driven approach** - Write tests for each incremental change
- **Continuous validation** - Ensure each step works before proceeding

### Pre-Push Quality Checks
- **ALWAYS run format and lint checks before pushing**
- Backend: `cd backend && poetry run ruff format --check src tests && poetry run ruff check src tests`
- Frontend: `cd frontend && npm run lint`
- **Run all tests** to ensure nothing is broken
- **Never push without passing quality checks**

## Development Loop

1. **Identify** the smallest meaningful change
2. **Implement** the minimal code needed
3. **Review** changes with user before commit
4. **Run quality checks** - format, lint, test
5. **Commit** only after explicit approval and passing checks
6. **Validate** the change works as expected
7. **Iterate** to the next small change

### Code Quality Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) Python style guide
- Use **Ruff** for linting and formatting
- Follow REST conventions for consistent API design
- Async test client from day 0
- Comprehensive API documentation with response models

## Commit Message Format

Follow [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Common Types
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `ci:` - CI/CD changes
- `build:` - Build system changes

### Examples
- `feat(search): add semantic video search`
- `fix(player): resolve timeline navigation bug`
- `docs: update installation instructions`
- `refactor(db): simplify video metadata schema`

## Quality Gates

- No code commits without user approval
- Each change must have clear purpose and scope
- Changes should not break existing functionality
- Maintain traceability to requirements and tasks
- All commits must follow conventional commit format

## Exception Handling

- If urgent fixes are needed, still follow approval process
- Document any deviations from normal workflow
- Retrospective review of emergency changes
## Communication Standards

### Summarization
- **Do not summarize work unless explicitly asked by the user**
- Avoid verbose recaps or lengthy summaries of completed tasks
- Keep responses focused and concise
- Only provide summaries when user requests them
