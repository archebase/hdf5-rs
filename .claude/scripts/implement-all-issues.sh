#!/usr/bin/env bash
#
# implement-all-issues.sh
# Two-phase workflow:
#   1. Planning: LLM analyzes issues and creates implementation order
#   2. Execution: Loop through plan, invoke /implement-issue for each
#
# This script is an ORCHESTRATOR only. Quality checks (PR review, coverage,
# Greptile, CI validation) are handled by /implement-issue command.
#

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

PLAN_FILE="${PLAN_FILE:-$HOME/.claude/implementation-plan.json}"
STATE_FILE="${STATE_FILE:-$HOME/.claude/implementation-state.json}"
MAX_ISSUES="${MAX_ISSUES:-}"           # Empty = unlimited
LABEL_FILTER="${LABEL_FILTER:-}"       # Only process issues with this label
ASSIGNEE_FILTER="${ASSIGNEE_FILTER:-}" # Only issues for this assignee ("none" = unassigned)
ITERATION_DELAY="${ITERATION_DELAY:-5}" # Seconds between issues

# ============================================================================
# LOGGING
# ============================================================================

LOG_COLOR="\033[0;32m"
WARN_COLOR="\033[0;33m"
ERROR_COLOR="\033[0;31m"
RESET_COLOR="\033[0m"

log() { echo -e "${LOG_COLOR}[INFO]${RESET_COLOR} $*"; }
log_warn() { echo -e "${WARN_COLOR}[WARN]${RESET_COLOR} $*"; }
log_error() { echo -e "${ERROR_COLOR}[ERROR]${RESET_COLOR} $*"; }
log_sep() { echo "─────────────────────────────────────────────────────────"; }

# ============================================================================
# USAGE
# ============================================================================

usage() {
    cat <<'EOF'
Usage: implement-all-issues.sh [OPTIONS]

Two-phase orchestrator for implementing GitHub issues.
Quality checks (PR review, coverage, Greptile, CI) are handled by /implement-issue.

PHASE 1 - Planning: LLM analyzes issues and creates implementation order
PHASE 2 - Execution: Execute each issue using /implement-issue command

OPTIONS:
    --repo <owner/name>     Repository (default: auto-detect from git origin)
    --label <label>         Only process issues with this label
    --assignee <username>   Only process issues for this assignee (use "none" for unassigned)
    --max-issues <n>        Stop after N issues (default: unlimited)
    --delay <seconds>       Delay between issues (default: 5)
    --plan-only             Only create plan, don't execute
    --execute-only          Skip planning, use existing plan file
    --fresh-plan            Ignore existing plan, create new one
    -h, --help              Show this help

ENVIRONMENT VARIABLES:
    PLAN_FILE               Where to store plan (default: ~/.claude/implementation-plan.json)
    STATE_FILE              Where to store state (default: ~/.claude/implementation-state.json)
    MAX_ISSUES              Maximum issues to process
    LABEL_FILTER            Label filter
    ASSIGNEE_FILTER         Assignee filter
    ITERATION_DELAY         Seconds between iterations

EXAMPLES:
    # Full workflow (plan + execute)
    implement-all-issues.sh

    # Create plan first, review it, then execute
    implement-all-issues.sh --plan-only
    # Review ~/.claude/implementation-plan.json
    implement-all-issues.sh --execute-only

    # Create fresh plan (ignore existing)
    implement-all-issues.sh --fresh-plan

    # Specific repo with filters
    implement-all-issues.sh --repo archebase/strata --label priority-high
EOF
    exit 0
}

# ============================================================================
# REPO DETECTION
# ============================================================================

detect_repo() {
    if [[ -n "${REPO:-}" ]]; then
        echo "$REPO"
        return
    fi

    local remote_url
    remote_url=$(git remote get-url origin 2>/dev/null || true)

    if [[ -z "$remote_url" ]]; then
        log_error "Could not detect repository. Use --repo or set REPO env var."
        exit 1
    fi

    # git@github.com:owner/repo.git
    if [[ "$remote_url" =~ git@github.com:([^/]+)/(.+).git ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        return
    fi

    # https://github.com/owner/repo.git
    if [[ "$remote_url" =~ https://github.com/([^/]+)/(.+).?git? ]]; then
        echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
        return
    fi

    log_error "Could not parse repo from: $remote_url"
    exit 1
}

# ============================================================================
# ISSUE FETCHING
# ============================================================================

fetch_issues() {
    local repo="$1"
    log "Fetching issues from $repo..."

    gh issue list --repo "$repo" --state open --limit 500 \
        --json number,title,body,labels,assignees,createdAt,url 2>/dev/null || {
        log_error "Failed to fetch issues. Check 'gh auth status'."
        exit 1
    }
}

# ============================================================================
# PHASE 1: PLANNING
# ============================================================================

create_plan() {
    local repo="$1"
    local issues_json="$2"

    log_sep
    log "PHASE 1: CREATING IMPLEMENTATION PLAN"
    log_sep

    # Create a temporary file with the issues data
    local issues_file
    issues_file=$(mktemp)
    echo "$issues_json" > "$issues_file"

    # Create prompt file
    local prompt_file
    prompt_file=$(mktemp)

    cat > "$prompt_file" <<EOF
You are a technical lead planning the implementation of GitHub issues.

## Repository
$repo

## Filter Settings
- Label filter: ${LABEL_FILTER:-none}
- Assignee filter: ${ASSIGNEE_FILTER:-none}
- Max issues: ${MAX_ISSUES:-unlimited}

## Open Issues
\`\`\`
$(cat "$issues_file" | jq -r '.[] | "#\(.number): \(.title)\n  Labels: \(.labels // [] | map(.name) | join(", ") // "none")\n  Assignees: \(.assignees // [] | map(.login) | join(", ") // "none")\n  Body: \(.body // "empty" | .[0:500])\n"')
\`\`\`

## Your Task

Create an implementation plan that:

1. **Parses dependencies** from issue bodies. Look for:
   - "depends on #N", "blocked by #N", "requires #N"
   - "[depends on: #N]", "[blocked by: #N]"
   - Task list items: "- [ ] #N"

2. **Detects circular dependencies** and breaks them intelligently

3. **Orders issues** by:
   - Dependencies first (topological sort)
   - Priority (critical > high > medium > low)
   - Creation date (older first within same priority)
   - Unassigned before assigned

4. **Skips** issues with these labels:
   - wontfix, blocked, on-hold, design-needed

5. **Handles external dependencies** - if an issue depends on #N where #N is NOT in the processing set, skip with note

## Output Format

Write ONLY a valid JSON object to stdout:

\`\`\`json
{
  "repository": "$repo",
  "generated_at": "ISO-8601 timestamp",
  "filters": {
    "label": "${LABEL_FILTER:-null}",
    "assignee": "${ASSIGNEE_FILTER:-null}",
    "max_issues": ${MAX_ISSUES:-null}
  },
  "summary": {
    "total_issues": <number of open issues>,
    "processable": <number after filtering>,
    "skipped": <number skipped>,
    "skipped_reasons": {
      "wontfix_label": <count>,
      "external_dependency": <count>
    }
  },
  "dependency_graph": {
    "nodes": [
      {"number": 42, "title": "...", "priority": "critical", "dependencies": []}
    ],
    "edges": [
      {"from": 36, "to": 42, "reason": "#36 depends on #42"}
    ]
  },
  "implementation_order": [
    {
      "position": 1,
      "issue_number": 42,
      "title": "...",
      "priority": "critical",
      "dependencies": [],
      "estimated_complexity": "low|medium|high",
      "rationale": "No dependencies, highest priority"
    }
  ],
  "warnings": []
}
\`\`\`

Output ONLY the JSON, nothing else.
EOF

    # Invoke Claude Code to create the plan
    log "Invoking Claude Code to create implementation plan..."

    if command -v claude &>/dev/null; then
        claude < "$prompt_file" > "$PLAN_FILE" || {
            log_error "Claude Code execution failed"
            rm -f "$prompt_file" "$issues_file"
            exit 1
        }
    else
        log_warn "Claude CLI not found. Creating basic plan..."

        # Fallback: basic plan with jq
        jq -n \
            --arg repo "$repo" \
            --argjson issues "$issues_json" \
            '{
                repository: $repo,
                generated_at: (now | todate),
                filters: {label: $LABEL_FILTER, assignee: $ASSIGNEE_FILTER, max_issues: $MAX_ISSUES},
                summary: {total_issues: ($issues | length), processable: ($issues | length), skipped: 0, skipped_reasons: {}},
                dependency_graph: {nodes: [], edges: []},
                implementation_order: ($issues | map({
                    position: (.number | tostring),
                    issue_number: .number,
                    title: .title,
                    priority: "medium",
                    dependencies: [],
                    estimated_complexity: "medium",
                    rationale: "Basic ordering"
                })),
                warnings: ["Basic plan - Claude CLI not available"]
            }' > "$PLAN_FILE"
    fi

    rm -f "$prompt_file" "$issues_file"

    # Validate plan
    if ! jq empty "$PLAN_FILE" 2>/dev/null; then
        log_error "Generated plan is not valid JSON"
        cat "$PLAN_FILE"
        exit 1
    fi

    log "Plan saved to: $PLAN_FILE"
}

display_plan() {
    if [[ ! -f "$PLAN_FILE" ]]; then
        log_error "No plan found at $PLAN_FILE"
        log "Run with --plan-only to create one first"
        exit 1
    fi

    log_sep
    log "IMPLEMENTATION PLAN"
    log_sep

    echo
    jq -r '
        "Repository: \(.repository)",
        "Generated: \(.generated_at)",
        "",
        "Summary:",
        "  Total issues: \(.summary.total_issues)",
        "  Processable: \(.summary.processable)",
        "  Skipped: \(.summary.skipped)",
        "",
        ((.warnings // []) | length > 0),
        "Warnings: \(.warnings | length)",
        ((.warnings // []) | map("  - \(.)") | join("\n"))
    ' "$PLAN_FILE"

    echo
    log_sep
    log "IMPLEMENTATION ORDER:"
    log_sep

    jq -r '
        .implementation_order[] |
        "\(.position | tostring | pad_left(3 | tostring; " ")). #\(.issue_number | tostring | pad_left(4 | tostring; " "))  \(.title)  [\(.priority)]\n   └─ deps: \(.dependencies | join(", ") // "none"), \(.estimated_complexity)"
    ' "$PLAN_FILE"

    log_sep
    echo
}

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo '{"completed": [], "failed": [], "current_position": 0}'
    fi
}

save_state() {
    local state="$1"
    echo "$state" > "$STATE_FILE"
}

clear_state() {
    rm -f "$STATE_FILE"
}

# ============================================================================
# PHASE 2: EXECUTION
# ============================================================================

# Check if a PR already exists for an issue's branch
check_existing_pr() {
    local repo="$1"
    local branch="$2"
    local issue_number="$3"

    local existing_pr
    existing_pr=$(gh pr list --repo "$repo" --head "$branch" --json number,state,title,body,mergedAt --jq '.[0]' 2>/dev/null || echo "null")

    if [[ "$existing_pr" == "null" ]]; then
        echo "none"
        return
    fi

    local pr_number pr_state pr_merged
    pr_number=$(echo "$existing_pr" | jq -r '.number // empty')
    pr_state=$(echo "$existing_pr" | jq -r '.state // "UNKNOWN"')
    pr_merged=$(echo "$existing_pr" | jq -r '.mergedAt // empty')

    # Case 1: PR was already merged
    if [[ -n "$pr_merged" ]]; then
        echo "merged:$pr_number"
        return
    fi

    # Case 2: PR is open/closed but not merged - check if it references this issue
    local pr_body
    pr_body=$(echo "$existing_pr" | jq -r '.body // ""')

    if echo "$pr_body" | grep -q "#$issue_number"; then
        echo "exists:$pr_number:$pr_state"
    else
        echo "different:$pr_number:$pr_state"
    fi
}

execute_plan() {
    local repo="$1"

    log_sep
    log "PHASE 2: EXECUTING IMPLEMENTATION PLAN"
    log_sep
    log "Quality checks (PR review, coverage, Greptile, CI) handled by /implement-issue"
    log_sep

    if [[ ! -f "$PLAN_FILE" ]]; then
        log_error "No plan found at $PLAN_FILE"
        exit 1
    fi

    # Load state
    local state
    state=$(load_state)
    local completed
    completed=$(echo "$state" | jq -r '.completed[]')

    # Get implementation order from plan
    local count
    count=$(jq '.implementation_order | length' "$PLAN_FILE")

    local max_issues="${MAX_ISSUES:-999999}"
    local processed=0

    for ((i=0; i<count; i++)); do
        [[ $processed -ge $max_issues ]] && break

        local item
        item=$(jq -c ".implementation_order[$i]" "$PLAN_FILE")

        local number title
        number=$(jq -r '.issue_number' <<< "$item")
        title=$(jq -r '.title' <<< "$item")

        # Skip if already completed
        if echo "$completed" | grep -q "^$number$"; then
            log "[$((i+1))/$count] #$number already completed, skipping"
            continue
        fi

        log_sep
        log "[$((i+1))/$count] Executing #$number: $title"
        log_sep

        # Check for existing PR before executing
        local expected_branch="feature/issue-$number"
        local pr_check
        pr_check=$(check_existing_pr "$repo" "$expected_branch" "$number")

        case "$pr_check" in
            merged:*)
                local pr_num="${pr_check#merged:}"
                log "PR #$pr_num already merged, marking #$number as complete"
                state=$(echo "$state" | jq --arg num "$number" '.completed += [$num | tonumber]')
                save_state "$state"
                continue
                ;;
            exists:*open*)
                local pr_num="${pr_check#exists:}"
                pr_num="${pr_num%:*}"
                log "PR #$pr_num already exists and is open for #$number"
                state=$(echo "$state" | jq --arg num "$number" '.completed += [$num | tonumber]')
                processed=$((processed + 1))
                log "✓ #$number marked complete (PR #$pr_num exists)"
                save_state "$state"
                continue
                ;;
            different:*)
                local pr_num="${pr_check#different:}"
                pr_num="${pr_num%:*}"
                log_warn "Branch $expected_branch has PR #$pr_num for a DIFFERENT issue (combined PR)"
                log "Updating PR #$pr_num to include #$number..."
                if gh pr edit "$pr_num" --repo "$repo" --body "$(
                    gh pr view "$pr_num" --repo "$repo" --json body -q '.body' 2>/dev/null
                    echo ""
                    echo "Closes #$number"
                )" 2>/dev/null; then
                    log "✓ Updated PR #$pr_num to reference #$number"
                    state=$(echo "$state" | jq --arg num "$number" '.completed += [$num | tonumber]')
                    processed=$((processed + 1))
                else
                    log_warn "Failed to update PR"
                    state=$(echo "$state" | jq --arg num "$number" '.failed += [$num | tonumber]')
                fi
                save_state "$state"
                continue
                ;;
        esac

        # No existing PR - delegate to /implement-issue
        if [[ -f "$HOME/.claude/commands/implement-issue.md" ]]; then
            log "Delegating to: /implement-issue $number"
            log "(handles: PR review, coverage, Greptile, CI, merge)"

            if Skill skill "implement-issue" args "$number"; then
                state=$(echo "$state" | jq --arg num "$number" '.completed += [$num | tonumber]')
                processed=$((processed + 1))
                log "✓ #$number completed"
            else
                state=$(echo "$state" | jq --arg num "$number" '.failed += [$num | tonumber]')
                log_warn "✗ #$number failed"
            fi

            save_state "$state"
        else
            log_error "No implement-issue command found at ~/.claude/commands/"
            exit 1
        fi

        # Delay before next
        if [[ $i -lt $((count - 1)) ]] && [[ $processed -lt $max_issues ]]; then
            log "Waiting ${ITERATION_DELAY}s before next issue..."
            sleep "$ITERATION_DELAY"
        fi
    done

    log_sep
    log "EXECUTION COMPLETE"
    log_sep
    log "Processed: $processed issues"
    log "Completed: $(echo "$state" | jq '.completed | length')"
    log "Failed: $(echo "$state" | jq '.failed | length')"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    local repo=""
    local plan_only=false
    local execute_only=false
    local fresh_plan=false

    # Parse args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo)        repo="$2"; shift 2 ;;
            --label)       LABEL_FILTER="$2"; shift 2 ;;
            --assignee)    ASSIGNEE_FILTER="$2"; shift 2 ;;
            --max-issues)  MAX_ISSUES="$2"; shift 2 ;;
            --delay)       ITERATION_DELAY="$2"; shift 2 ;;
            --plan-only)   plan_only=true; shift ;;
            --execute-only) execute_only=true; shift ;;
            --fresh-plan)  fresh_plan=true; shift ;;
            -h|--help)     usage ;;
            *) log_error "Unknown: $1"; usage ;;
        esac
    done

    # Detect repo
    repo=$(detect_repo)
    log "Repository: $repo"

    # Execute only?
    if [[ "$execute_only" == "true" ]]; then
        display_plan
        execute_plan "$repo"
        exit 0
    fi

    # Fresh plan?
    if [[ "$fresh_plan" == "true" ]]; then
        rm -f "$PLAN_FILE"
        clear_state
        log "Cleared existing plan and state"
    fi

    # Check if plan exists
    if [[ -f "$PLAN_FILE" ]] && [[ "$fresh_plan" != "true" ]]; then
        log "Found existing plan: $PLAN_FILE"
        log "Use --fresh-plan to regenerate"
        display_plan
    else
        # Fetch issues
        local issues_json
        issues_json=$(fetch_issues "$repo")

        local issue_count
        issue_count=$(echo "$issues_json" | jq 'length')
        log "Found $issue_count open issues"

        # Create plan
        create_plan "$repo" "$issues_json"

        # Display plan
        display_plan
    fi

    # Plan only?
    if [[ "$plan_only" == "true" ]]; then
        log "Plan-only mode - not executing"
        log "Run again with --execute-only to implement"
        exit 0
    fi

    # Execute
    execute_plan "$repo"
}

main "$@"
