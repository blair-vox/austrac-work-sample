# AUSTRAC Work Sample

This repository documents data science, machine learning, and engineering work across four VoxPopAI repositories. The work includes implementations of synthetic population generation, vector embeddings, LLM agent systems, API design, and privacy-preserving data processing.

## Overview

This work sample provides a comprehensive view of data science and programming capabilities through four interconnected repositories:

1. **VoxPop-SynC-GUI**: Synthetic population generation pipeline using SynC methodology (Gaussian Copula) with neural networks
2. **VoxPopAI-KB**: Knowledge base service with synthetic persona management, vector embeddings, and LLM-enhanced agents
3. **voxpop-crew**: Synthetic population simulation system with geographic visualization and CrewAI-based agents
4. **doctor-okso**: Privacy-preserving email processing and knowledge base system with semantic search

All data referenced in these repositories is **synthetic or publicly available**. No sensitive, proprietary, or personally identifiable information is included in this documentation or the referenced codebases.

## Data Science Focus Areas

This work sample includes implementations of:

- **Synthetic Population Generation**: Creating and managing large-scale synthetic populations (5.2M+ personas) from census data
- **Vector Embeddings & Retrieval**: Implementing semantic search using pgvector and OpenAI embeddings
- **LLM Agent Systems**: Building multi-agent systems with planning, reasoning, retrieval, and critique capabilities
- **Privacy-Preserving Data Processing**: Implementing PII redaction and privacy gates for sensitive data
- **API Design**: Creating FastAPI endpoints that expose data science functionality
- **Geographic Visualization**: Mapping and visualizing synthetic populations across Australian Statistical Areas

## Repository Structure

```
austrac-work-sample/
├── README.md                    # This file - main entry point
├── WORK_SAMPLE_OVERVIEW.md      # Technical deep-dive documentation
├── links/
│   └── VOXPOP_REPOS.md          # Links to repositories and key files
└── dashboards/                  # Visualization documentation and examples
    ├── visualization_overview.md  # Visualization capabilities documentation
    ├── example_plots.md          # Example visualization plots
    └── *.png                     # Example plot images
```

## Navigation Guide

### For AUSTRAC Reviewers

1. **Start here**: This README provides an overview of the work sample
2. **Technical details**: See [WORK_SAMPLE_OVERVIEW.md](WORK_SAMPLE_OVERVIEW.md) for in-depth explanations of data science components
3. **Code references**: See [links/VOXPOP_REPOS.md](links/VOXPOP_REPOS.md) for direct links to repositories and key files
4. **Visualizations**: See [dashboards/example_plots.md](dashboards/example_plots.md) for example visualization plots and [dashboards/visualization_overview.md](dashboards/visualization_overview.md) for documentation

### Key Documentation Files

- **[WORK_SAMPLE_OVERVIEW.md](WORK_SAMPLE_OVERVIEW.md)**: Comprehensive technical documentation covering:
  - Synthetic population generation methodology
  - Vector embedding strategies
  - LLM agent architectures
  - Privacy-preserving data processing
  - API design patterns
  - Cross-repository patterns and principles

- **[links/VOXPOP_REPOS.md](links/VOXPOP_REPOS.md)**: Reference document with:
  - GitHub repository links
  - Key data science files and their purposes
  - Main entry points for each repository
  - Important modules and components

## Privacy and Governance

This work sample includes privacy-preserving techniques and governance practices:

- **PII Redaction**: Privacy gates automatically redact personally identifiable information
- **Privacy-First Design**: Contact extraction and data processing include vetting mechanisms
- **User-Scoped Data**: All data is isolated by user/region with appropriate access controls

These practices include privacy protection, data protection mechanisms, and responsible data handling.

## Linked Repositories

The following repositories contain the actual code implementations:

1. **VoxPop-SynC-GUI** - Synthetic population generation pipeline (SynC methodology)
2. **VoxPopAI-KB** - Knowledge base and agent service
3. **voxpop-crew** - Synthetic population simulation system
4. **doctor-okso** - Privacy-preserving knowledge base system

See [links/VOXPOP_REPOS.md](links/VOXPOP_REPOS.md) for GitHub URLs and detailed file references.

## Quick Start for Reviewers

1. Read this README for context
2. Review [WORK_SAMPLE_OVERVIEW.md](WORK_SAMPLE_OVERVIEW.md) for technical details
3. Explore linked repositories via [links/VOXPOP_REPOS.md](links/VOXPOP_REPOS.md)
4. Examine code files referenced in the documentation

## Contact

For questions about this work sample, please refer to the application materials or contact Blair Conn.

---

**Note**: This repository contains documentation only. All code implementations are in the linked repositories.
