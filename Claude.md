# Strata - Embodiment AI Query Engine

This is the Strata project, a domain-specific query engine for embodiment AI data.

## Project Structure

```
strata/
├── api/                    # Go API layer (Fiber framework)
│   ├── cmd/strata-api/     # Main entry point
│   └── internal/           # Internal packages (handlers, middleware, services)
├── engine/                 # Query engine (Rust + Python)
│   ├── strata-core/        # Rust core (DataFusion extensions)
│   ├── strata-flight/      # Arrow Flight SQL server and CLI
│   ├── strata-python/      # PyO3 Python bindings
│   └── strata/             # Python package
├── catalog/                # MySQL migrations
└── deploy/                 # Docker, K8s configs
```

## Key Technologies

- **DataFusion (Rust)**: Core query engine with SQL parsing, optimization, execution
- **Arrow Flight SQL**: High-performance gRPC protocol for database connectivity (port 50051)
- **PyO3**: Rust-Python bindings for the query engine
- **Go (Fiber)**: REST API layer with auth, rate limiting
- **MySQL**: Catalog for dataset metadata, UDFs, query logs
- **Lance**: Columnar format with vector search for embeddings
- **Ray**: Distributed execution for materialization and Python UDF parallelization

## Design Documentation

Design documents live in `design/`. These describe what we're building and why—not how to use the system.

```
design/
├── README.md                    # Design doc index
├── DEVELOPMENT.md               # Developer setup guide
│
├── data_ingestion/              # Data ingestion feature
│   ├── overview.md              # Core concepts, SQL syntax
│   ├── bag_support.md           # Phase 3c: BAG design
│   ├── ray_integration.md       # Phase 3d: Ray design
│   └── status.md                # Implementation status, task lists
│
└── distributed_query/           # Distributed query feature
    └── strategy.md
```

### Design Doc Conventions

1. **One feature per folder** — Group related design docs together (e.g., `data_ingestion/`, `distributed_query/`)

2. **Separate design from status** — Design docs describe architecture; `status.md` tracks implementation progress

3. **Feature-scoped status** — Each feature folder has its own `status.md` with:
   - Implementation phases with checkboxes
   - Component status table
   - Task lists

4. **Use `design/` for internal docs** — User-facing documentation (tutorials, how-to guides) goes in `docs/` (future)

### Writing a Design Doc

Start with a clear problem statement and proposed solution:

```markdown
# Feature Name Design

## Problem Statement
What problem are we solving?

## Proposed Solution
High-level architecture and approach.

## Detailed Design
- Component diagrams
- Data structures
- API contracts

## Implementation Plan
Phase 1, Phase 2, ...

## Open Questions
Unresolved issues for discussion
```

Reference existing design docs for style:
- `design/data_ingestion/overview.md` — Broad feature overview
- `design/data_ingestion/bag_support.md` — Specific component design
- `design/distributed_query/strategy.md` — Strategic analysis

## Development Commands

```bash
make dev-up              # Start infrastructure (MySQL, MinIO, Ray)
make build               # Build all components
make test                # Run all tests
make run-api             # Start the API server
make run-flight-server   # Start Arrow Flight SQL server (port 50051)
make strata-cli          # Run CLI client (ARGS="--query 'SELECT 1'")
make help                # Show all commands
```

## Claude Code Guidelines

When working on this project:

1. **API Changes**: Go code in `api/`. Use Fiber conventions, add handlers to `internal/handler/`.
2. **Query Engine**: Rust code in `engine/strata-core/`. DataFusion extensions, UDFs, table providers.
3. **Flight SQL**: Rust code in `engine/strata-flight/`. Server, session management, CLI client.
4. **Python Bindings**: `engine/strata-python/` for PyO3 bindings, `engine/strata/` for Python API.
5. **Database**: Add migrations to `catalog/migrations/` with sequential numbering.

### Code Style

- **Rust**: Use `cargo fmt` and `cargo clippy`
- **Python**: Use `black` and `ruff`
- **Go**: Use `gofmt` and standard Go conventions

#### Example Naming Convention

In code comments, documentation, and examples, use the consistent naming sequence:
- **foo** - First example item
- **bar** - Second example item
- **baz** - Third example item
- **pux** - Fourth example item (used when a fourth distinct example is needed)

This convention applies to:
- Variable names in code comments
- Type names in documentation examples
- Function/method names in examples
- File names in examples

```
// Good: Uses standard example names
let foo_type = "foo/Msg";
let bar_type = "bar/Msg";
let baz_type = "baz/Msg";
let pux_type = "pux/Msg";

// Bad: Uses inconsistent naming
let alpha = "alpha/Msg";
let beta = "beta/Msg";
let gamma = "gamma/Msg";
```

### Testing

#### Test Levels

1. **Unit Tests**: Test individual components in isolation
   - Located in `tests/<component>_tests.rs` files
   - Fast, no external dependencies
   - Example: `ros1_decoder_tests.rs`, `topic_mapper_tests.rs`

2. **Integration Tests**: Test component interactions using direct library calls
   - Located in `tests/<component>_integration_tests.rs` or `tests/<component>_converter_tests.rs`
   - May use `StrataSession` directly
   - Example: `bag_converter_tests.rs`, `mcap_integration_tests.rs`

3. **E2E Tests**: Test the full system through network protocols
   - Located in `strata-flight/tests/strata_sql_tests.rs` (Rust) or `tests/e2e/` (Python)
   - Start Flight SQL server as a separate process
   - Use `strata-cli` or Flight SQL client to execute commands
   - Run with `make test-e2e`

#### Test Naming Convention

Use descriptive names that clearly state what is being tested and the expected behavior:

```
test_<subject>_<behavior_or_condition>
```

Patterns by test type:
- **Success cases**: `test_<component>_<action>_<input_or_condition>`
  - `test_reader_opens_valid_bag_file`
  - `test_decoder_creates_successfully`

- **Error cases**: `test_<component>_<action>_for_<error_condition>`
  - `test_reader_returns_error_for_nonexistent_file`
  - `test_decode_without_schema_returns_error`

- **Mapping/transformation**: `test_<component>_maps_<input>_to_<output>`
  - `test_mapper_maps_joint_state_to_joint_states_stream`
  - `test_mapper_maps_image_to_video_frames_stream`

- **Property assertions**: `test_<component>_<property>_<assertion>`
  - `test_reader_messages_have_valid_timestamps`
  - `test_reader_connections_contains_valid_metadata`

Examples:
- `test_reader_opens_valid_bag_file` - Good: clear subject and behavior
- `test_decode_int32_returns_correct_field` - Good: specific input and outcome
- `test_bag_reader` - Bad: too vague, doesn't describe what's being tested

#### Test File Organization

```
engine/strata-core/tests/
├── common/
│   └── mod.rs                # Shared utilities, fixtures, assertions
├── ros1_decoder_tests.rs     # Unit tests for Ros1Decoder
├── bag_reader_tests.rs       # Unit tests for BagReader
├── bag_converter_tests.rs    # Integration tests for BAG conversion
├── topic_mapper_tests.rs     # Unit tests for TopicMapper
├── mcap_tests.rs             # Unit tests for MCAP reader
├── mcap_integration_tests.rs     # Integration tests for MCAP conversion
└── session_integration_tests.rs  # StrataSession integration tests

engine/strata-flight/tests/
├── common/
│   └── mod.rs                # Flight SQL test utilities
├── query_tests.rs            # Flight SQL query tests
├── metadata_tests.rs         # Flight SQL metadata tests
├── prepared_stmt_tests.rs    # Prepared statement tests
├── transaction_tests.rs      # Transaction handling tests
├── doput_tests.rs            # DoPut operation tests
├── tls_tests.rs              # TLS/security tests
└── strata_sql_tests.rs       # Strata SQL syntax E2E tests (BAG/MCAP)

tests/e2e/
└── test_e2e.py               # Python E2E tests using subprocess
```

#### Shared Test Utilities

Use the `common` module for shared test utilities:

```rust
mod common;

#[test]
fn test_example() {
    let bag_path = common::bag_demo_fixture();
    skip_if_missing!(&bag_path, "demo.bag");

    // Use common assertions
    common::assert_lance_dataset_valid(&output_path);
}
```

Available utilities:
- `fixtures_dir()`, `bag_demo_fixture()`, `mcap_nissan_fixture()` - Fixture paths
- `temp_output_dir()`, `temp_file_with_content()` - Temporary files
- `assert_lance_dataset_valid()`, `assert_episodes_subdataset_exists()` - Lance assertions
- `assert_error_contains()`, `assert_is_error()` - Error assertions
- `default_bag_convert_options()`, `default_mcap_convert_options()` - Default options
- `Ros1MessageBuilder` - Build test message data

#### Writing Good Assertions

Always include context in assertions:

```rust
// Bad - no context on failure
assert!(result.is_ok());
assert!(count > 0);

// Good - clear failure message
assert!(
    result.is_ok(),
    "Expected successful conversion, got error: {:?}",
    result.err()
);
assert!(
    count > 0,
    "Expected at least one message, got {}",
    count
);
```

#### Running Tests

```bash
make test                 # Run all tests
make test-engine          # Run Rust engine tests only
cd engine && cargo test   # Run Rust tests with output
cd engine && cargo test -- --nocapture  # Show println! output
make test-e2e             # Run E2E tests (starts Flight server)
```

### Task Management with Todo Lists

For complicated tasks involving multiple components or phases, always use todo lists to track progress:

1. **When to Create a Todo List**:
   - Multi-file changes spanning different modules
   - Implementation of design documents with multiple phases
   - Bug fixes requiring investigation across components
   - Any task with 3+ distinct steps

2. **Todo List Structure**:
   - Group items by logical phases or components
   - Use `===` prefix for phase headers (e.g., `=== Phase 1: Foundation ===`)
   - Mark status: `completed`, `in_progress`, or `pending`
   - Only one item should be `in_progress` at a time

3. **Maintaining the Todo List**:
   - Update status immediately when completing a task
   - Add new items discovered during implementation
   - Remove items that become irrelevant
   - Keep the list visible to track overall progress

4. **Example Todo List for Multi-Phase Implementation**:
   ```
   === Phase 1: Core Infrastructure ===
   [completed] Create base types and traits
   [completed] Implement worker abstraction
   [in_progress] Add progress tracking

   === Phase 2: Integration ===
   [pending] Wire up to existing API
   [pending] Add SQL command support

   === Testing ===
   [pending] Unit tests
   [pending] Integration tests
   ```

### Debug Scripts

**RULE: All debug/diagnostic scripts go in `src/bin/`, not inline Python/bash scripts.**

When investigating issues or creating diagnostic tools:
- Create proper Rust binaries in `src/bin/*.rs`
- Use descriptive names: `check_*.rs`, `debug_*.rs`, `diagnose_*.rs`
- Build and run with `cargo run --bin <name>` or `cargo build --bin <name>`
- This ensures debug tools are version-controlled, type-checked, and reusable

**Examples**:
- `src/bin/mcap_info.rs` - Dump MCAP file info
- `src/bin/debug_schema.rs` - Examine schema parsing
- `src/bin/check_bag.rs` - Verify BAG file structure

**When to create debug scripts**:
- Inspecting MCAP/BAG file contents
- Verifying schema transformations
- Tracing decoder behavior
- Any investigation that benefits from a reusable tool

**Anti-pattern to avoid**:
```bash
# DON'T: Use inline Python/bash heredocs
cat > /tmp/check.py << EOF
import...
EOF
python /tmp/check.py

# DO: Create a proper binary in src/bin/
cargo run --bin check_mcap
```
