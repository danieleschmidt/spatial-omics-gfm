# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Spatial-Omics GFM project. ADRs document important architectural decisions, their context, and consequences.

## ADR Format

Each ADR follows this template:

```markdown
# ADR-XXXX: [Title]

**Date**: YYYY-MM-DD  
**Status**: [Proposed | Accepted | Deprecated | Superseded]  
**Deciders**: [List of people involved in decision]

## Context

[Describe the forces at play, including technological, political, social, and project local]

## Decision

[Describe our response to these forces]

## Consequences

[Describe the resulting context, after applying the decision]

## Alternatives Considered

[List other options that were considered]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](./0001-graph-transformer-architecture.md) | Graph Transformer Architecture Choice | Accepted | 2025-08-02 |

## Creating New ADRs

1. Copy the template above
2. Assign the next sequential number
3. Use descriptive kebab-case titles
4. Include all stakeholders in the decision process
5. Update this index when adding new ADRs

## ADR Lifecycle

- **Proposed**: Under discussion
- **Accepted**: Decision made and implemented
- **Deprecated**: No longer relevant but kept for historical context
- **Superseded**: Replaced by a newer ADR