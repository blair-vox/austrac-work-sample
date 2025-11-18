# VoxPopAI Repository Links and Key Files

This document provides direct links to the four VoxPopAI repositories and identifies key data science files within each repository.

## Repository 1: VoxPop-SynC-GUI

### GitHub Repository
**URL**: https://github.com/blair-vox/VoxPop-SynC-GUI

### Overview
Synthetic population generation pipeline using SynC methodology (Gaussian Copula) with neural networks. This is the foundational system that generates synthetic personas from aggregated census data.

### Key Data Science Files

#### Pipeline Orchestration
- **`main.py`**
  - Command-line interface for the SynC pipeline
  - YAML configuration loading and validation
  - Parameter override capabilities
  - Entry point for single-location and all-locations processing

- **`models/synthetic_populations.py`**
  - Main pipeline orchestration
  - Single location processing (`main()`)
  - All-locations processing with error handling (`process_all_locations()`)
  - Progress tracking and resume capability
  - Output file generation
  - Logging and error recovery

#### Core Statistical & ML Functions
- **`models/functions.py`**
  - `process_file()`: Processes demographic files and generates synthetic data
  - `model_to_pop()`: Neural network model for population prediction
  - `sample_data()`: Generates synthetic samples from correlation structure (Gaussian Copula)
  - `match_marginal()`: Matches synthetic data to target marginals (iterative algorithm)
  - `assign_persona_based_features()`: Adds business logic features
  - `generate_big_five_traits()`: Adds personality traits
  - `to_dummy()`: One-hot encoding for categorical variables
  - `power_two()`: Helper for neural network layer sizing
  - `sampling()`: Categorical distribution sampling

#### Configuration & Data Management
- **`config.yaml`** / **`config_example.yaml`**
  - YAML configuration files defining:
    - Project metadata
    - Data source file paths
    - Geographic filtering parameters
    - Feature specifications (columns and names)
    - Processing parameters (seeds, verbose mode)
    - Output settings

- **`models/abs_data_parser.py`**
  - ABS (Australian Bureau of Statistics) data parsing
  - Census data processing utilities

- **`models/correlated_features.py`**
  - Feature correlation analysis
  - Dependency structure modeling

#### Diagnostic & Validation Tools
- **`check_population_issues.py`**
  - Validates output synthetic populations
  - Checks for population size mismatches
  - Identifies data quality issues

- **`check_core_duplicates.py`**
  - Validates core data input
  - Checks for duplicate records
  - Data quality assurance

#### Batch Processing
- **`run_all_locations.py`**
  - Convenience script for all-locations processing
  - Wrapper around main pipeline
  - Error handling and logging

#### Testing & Development
- **`test_big_five.py`**
  - Tests Big 5 personality trait generation
  - Validates trait assignment logic

- **`test_postprocessing.py`**
  - Tests post-processing steps
  - Validates feature generation

- **`test_location_column.py`**
  - Tests geographic filtering
  - Validates location column processing

#### Legacy R Implementation
- **`models/functions.R`**
  - Original R implementation of SynC functions
  - Reference implementation

- **`models/main.R`**
  - R script for synthetic population generation
  - Legacy pipeline

- **`models/synthetic_population.R`**
  - R implementation of SynC pipeline
  - Complete workflow in R

### Main Entry Points
- **Pipeline Execution**: `main.py` (run with `python main.py --config <config_file>`)
- **All Locations**: `run_all_locations.py` (convenience script)
- **R Pipeline**: `models/synthetic_population.R` (legacy)

### Documentation
- **`README.md`**: Comprehensive documentation with quick start guide
- **`docs/synthetic_population_explained.md`**: Detailed explanation of SynC methodology
- **`docs/architecture_overview.md`**: System architecture documentation
- **`docs/configuration_guide.md`**: Configuration file guide
- **`docs/troubleshooting_guide.md`**: Troubleshooting and diagnostics

### Key Features
- SynC methodology implementation (Gaussian Copula)
- Neural network-based feature prediction
- Configuration-driven pipeline (YAML)
- All-locations processing with error handling
- Resume capability for interrupted runs
- Comprehensive logging and progress tracking
- 39+ attributes per synthetic individual
- Big 5 personality traits
- Behavioral feature generation

---

## Repository 2: VoxPopAI-KB

### GitHub Repository
**URL**: https://github.com/blair-vox/VoxpopAI-KB

### Overview
Knowledge base service with synthetic persona management, vector embeddings, and LLM-enhanced agents.

### Key Data Science Files

#### Persona Management
- **`kb_service/personas/populate_index.py`**
  - Populates PostgreSQL index from Parquet files
  - Processes 5.2M+ personas across 415+ files
  - Implements incremental updates using file hashing
  - Maps 44 persona attributes to database columns

- **`kb_service/personas/build_persona_ids.py`**
  - Generates deterministic UUIDs for personas
  - Handles persona ID assignment

- **`kb_service/personas/abs_sa2_mapper.py`**
  - Maps personas to Australian Statistical Areas (SA2)
  - Geographic data processing

- **`kb_service/personas/location_mapping.py`**
  - Location-based persona mapping
  - Geographic hierarchy handling

#### Vector Embeddings & Retrieval
- **`kb_service/ingest/embeddings_local.py`**
  - Local embedding generation using SentenceTransformers
  - Batch processing for efficiency
  - Fallback to dummy vectors if model unavailable

- **`kb_service/agents/retriever.py`**
  - Advanced retrieval system with multiple strategies
  - Vector similarity search using pgvector
  - Keyword-based search using PostgreSQL FTS
  - Hybrid retrieval combining vector and keyword
  - Multi-stage retrieval with re-ranking

- **`kb_service/ingest/upsert_pgvector.py`**
  - Upserts embeddings to pgvector-enabled PostgreSQL
  - Vector storage and indexing

#### LLM Agent System
- **`kb_service/agents/planner.py`**
  - Intelligent planning using LLM
  - Generates dynamic filters for retrieval
  - Selects exemplars for guidance
  - Determines optimal chunk limits

- **`kb_service/agents/reasoner.py`**
  - Context-aware answer generation using LLM
  - Builds comprehensive context from persona and chunks
  - Extracts citations and reasoning chains
  - Assesses answer confidence

- **`kb_service/agents/critic.py`**
  - Quality critique of answers
  - Validates persona alignment
  - Assesses completeness and accuracy

- **`kb_service/agents/distiller.py`**
  - Response refinement and summarization
  - Removes redundancy
  - Formats for presentation

- **`kb_service/agents/runner.py`**
  - Orchestrates agent pipeline execution
  - Coordinates planner, retriever, reasoner, critic, distiller

- **`kb_service/agents/llm_client.py`**
  - LLM client wrapper
  - Handles API calls and error handling

#### Persona Hydration
- **`kb_service/hydrate/hydrate.py`**
  - Combines canonical, regional, and memory data
  - Creates unified persona profiles
  - Assigns display names

- **`kb_service/hydrate/knowledge_join.py`**
  - Joins regional knowledge packs with personas
  - Merges metrics and derived data

- **`kb_service/hydrate/memory_store.py`**
  - Manages persona memory (traits, stances, interactions)
  - Updates interaction statistics
  - Stores persona notes

#### Knowledge Base Management
- **`kb_service/kb/knowledge_packs.py`**
  - Manages regional knowledge packs
  - Stores metrics, derived data, and sources
  - Maps knowledge packs by region_code

#### Data Ingestion
- **`kb_service/ingest/ingest_csv.py`**
  - Ingests CSV files into knowledge base
  - Summarizes CSV content to text

- **`kb_service/ingest/ingest_txt.py`**
  - Ingests text files into knowledge base
  - Chunks text for embedding

- **`kb_service/ingest/ingest_pdf.py`**
  - Ingests PDF files into knowledge base
  - Extracts text from PDFs

- **`kb_service/ingest/ingest_html.py`**
  - Ingests HTML files into knowledge base
  - Extracts text from HTML

- **`kb_service/ingest/chunking.py`**
  - Text chunking strategies
  - Word-based chunking with overlap

#### API Endpoints
- **`kb_service/api/main.py`**
  - FastAPI application with all endpoints
  - Persona selection, hydration, agent execution
  - Knowledge pack management
  - Memory store operations

- **`kb_service/api/schemas.py`**
  - Pydantic models for request/response validation
  - Type definitions for API contracts

#### Database
- **`kb_service/db/ddl.sql`**
  - Database schema definitions
  - Tables for personas, chunks, knowledge packs, memory

- **`kb_service/db/persona_index.py`**
  - Persona index query functions
  - Filter options and metadata retrieval

- **`kb_service/db/session.py`**
  - Database session management
  - Connection pooling

#### Configuration
- **`kb_service/config/settings.py`**
  - Configuration management
  - Settings loading from YAML

### Main Entry Points
- **API Server**: `kb_service/api/main.py` (run with `uvicorn kb_service.api.main:app`)
- **Persona Population**: `kb_service/personas/populate_index.py` (command-line script)
- **Ingestion**: `kb_service/ingest/ingest_*.py` (command-line scripts)

### Documentation
- **`docs/README.md`**: Comprehensive documentation index
- **`docs/8_PERSONA_INDEX_POPULATION.md`**: Persona index population guide
- **`docs/6_MEMORY_STORE.md`**: Memory store documentation
- **`docs/9_ENHANCED_AGENTS.md`**: Enhanced agents documentation

---

## Repository 3: voxpop-crew

### GitHub Repository
**URL**: https://github.com/blair-vox/voxpop-crew

### Overview
Synthetic population simulation system with geographic visualization and CrewAI-based agents.

### Key Data Science Files

#### Persona Management
- **`crew_agents/voxpop_agents/personas.py`**
  - Loads personas from CSV/Parquet
  - Persona data transformation

- **`voxpopai/backend/agents/persona_builder.py`**
  - Persona construction logic
  - Persona attribute generation

#### Geographic Visualization
- **`geo_tools/plot_synthetic.py`**
  - Visualization functions for synthetic personas
  - Plot selected regions, personas on maps
  - Multi-region comparisons
  - Demographic heatmaps
  - Highlighted persona regions

- **`geo_tools/geo_utils.py`**
  - Geographic utility functions
  - SA2 shapefile loading
  - Location search

- **`geo_tools/location_search.py`**
  - Location search functionality
  - Geographic queries

- **`geo_tools/get_persona_locations.py`**
  - Extracts persona locations
  - Geographic data extraction

- **`geo_tools/config.py`**
  - Configuration for plotting
  - Style settings

#### CrewAI Agents
- **`crew_agents/voxpop_agents/crew.py`**
  - CrewAI crew definition
  - Agent orchestration setup

- **`crew_agents/voxpop_agents/enhanced_agents.py`**
  - Enhanced agent implementations
  - Agent capabilities

- **`crew_agents/voxpop_agents/enhanced_main.py`**
  - Main execution for enhanced agents
  - Agent workflow

- **`crew_agents/voxpop_agents/enhanced_tasks.py`**
  - Task definitions for agents
  - Task orchestration

- **`crew_agents/voxpop_agents/rag.py`**
  - RAG (Retrieval-Augmented Generation) implementation
  - Knowledge retrieval for agents

- **`crew_agents/voxpop_agents/style_expert_rag.py`**
  - Style expert RAG implementation
  - Specialized retrieval

- **`crew_agents/voxpop_agents/structured_memory.py`**
  - Structured memory for agents
  - Memory management

- **`crew_agents/voxpop_agents/memory.py`**
  - Memory implementation
  - Conversation memory

- **`crew_agents/voxpop_agents/prompts/persona_prompt.py`**
  - Persona prompt templates
  - LLM prompt engineering

#### Focus Group Facilitation
- **`voxpopai/backend/agents/focus_group_facilitator/facilitator_agent.py`**
  - Main facilitator agent class
  - Interview conduction logic

- **`voxpopai/backend/agents/focus_group_facilitator/interview_styles.py`**
  - Interview style implementations
  - Exploratory, laddering, narrative styles

- **`voxpopai/backend/agents/focus_group_facilitator/memory_manager.py`**
  - Conversation memory management
  - Session tracking

#### Simulation & Response
- **`voxpopai/backend/agents/crewai_simulation_agent.py`**
  - CrewAI-based simulation agent
  - Response simulation using CrewAI

- **`voxpopai/backend/agents/response_simulator.py`**
  - Response simulation logic
  - Persona response generation

#### Survey & Question Management
- **`voxpopai/backend/agents/survey_composer.py`**
  - Survey composition logic
  - Question generation

- **`voxpopai/backend/agents/question_critic.py`**
  - Question quality critique
  - Question evaluation

- **`voxpopai/backend/agents/crewai_question_critic.py`**
  - CrewAI-based question critic
  - Enhanced question evaluation

#### Analysis & Statistics
- **`voxpopai/backend/utils/stats.py`**
  - Statistical analysis functions
  - Demographic statistics
  - Location frequency analysis

- **`voxpopai/backend/utils/theme_categoriser.py`**
  - Theme categorization
  - Multi-class classification
  - Theme derivation

- **`voxpopai/backend/utils/driver_extractor.py`**
  - Driver extraction from responses
  - Key factor identification

- **`voxpopai/backend/utils/raker.py`**
  - RAKE (Rapid Automatic Keyword Extraction)
  - Keyword extraction

#### API Endpoints
- **`voxpopai/backend/app.py`**
  - FastAPI application
  - Main application setup

- **`voxpopai/backend/routers/personas.py`**
  - Persona API endpoints
  - Simulation endpoints
  - Persona selection

- **`voxpopai/backend/routers/focus_groups.py`**
  - Focus group API endpoints
  - Interview management

- **`voxpopai/backend/routers/surveys.py`**
  - Survey API endpoints
  - Survey composition and analysis

- **`voxpopai/backend/routers/question.py`**
  - Question API endpoints
  - Question critique

- **`voxpopai/backend/routers/geo.py`**
  - Geographic API endpoints
  - Location queries

- **`voxpopai/backend/routers/kb_filters.py`**
  - Knowledge base filter endpoints
  - Filter management

#### Utilities
- **`voxpopai/backend/utils/llm_wrapper.py`**
  - LLM client wrapper
  - API call handling

- **`voxpopai/backend/utils/mappings.py`**
  - Data mapping utilities
  - Transformation functions

- **`voxpopai/backend/utils/run_storage.py`**
  - Run storage utilities
  - S3 integration

- **`voxpopai/backend/utils/run_logger.py`**
  - Run logging utilities
  - Logging functionality

### Main Entry Points
- **API Server**: `voxpopai/backend/app.py` (run with `uvicorn voxpopai.backend.app:app`)
- **CrewAI Agents**: `crew_agents/voxpop_agents/enhanced_main.py`
- **Geographic Tools**: `geo_tools/cli.py` (command-line interface)

### Documentation
- **`CREWAI_INTEGRATION_GUIDE.md`**: CrewAI integration documentation
- **`BATCH_RUN_README.md`**: Batch run documentation

---

## Repository 4: doctor-okso

### GitHub Repository
**URL**: https://github.com/VoxPopAI/doctor-okso

### Overview
Privacy-preserving knowledge base system for processing school newsletters with semantic search.

### Key Data Science Files

#### Email Processing
- **`supabase/functions/process-emails-v2/index.ts`**
  - Main email processing function
  - Gmail API integration
  - HTML extraction and parsing
  - Newsletter processing

- **`supabase/functions/process-emails/index.ts`**
  - Legacy email processing (v1)
  - Email fetching and processing

#### Privacy & PII Redaction
- **`supabase/functions/_shared/privacy-gate.ts`**
  - Privacy gate implementation
  - PII redaction (emails, phones, URLs, names)
  - Section filtering (drop sensitive, keep useful)
  - Contact extraction

- **`supabase/functions/_shared/contacts.ts`**
  - Contact management utilities
  - Contact vetting logic
  - Contact hydration

#### Knowledge Base
- **`supabase/functions/process-knowledge-base-upload/index.ts`**
  - Knowledge base upload processing
  - Document parsing (PDF, DOCX, TXT, HTML)
  - URL extraction and classification
  - Chunking and embedding generation
  - Vector storage

- **`supabase/functions/reprocess-knowledge-base/index.ts`**
  - Reprocess knowledge base content
  - Update embeddings

#### AI Assistant & Search
- **`supabase/functions/okso-assistant/index.ts`**
  - AI assistant main function
  - Multi-strategy search (semantic, keyword, date, link)
  - AI-powered query routing
  - Answer generation with citations

#### Calendar Processing
- **`supabase/functions/analyze-calendar-image/index.ts`**
  - OCR and AI analysis of calendar images
  - Event extraction from images

- **`supabase/functions/consolidate-calendar-events/index.ts`**
  - Calendar event deduplication
  - Event consolidation

- **`supabase/functions/compare-calendar-events/index.ts`**
  - Calendar event comparison
  - Duplicate detection

- **`supabase/functions/generate-calendar-file/index.ts`**
  - Generate .ics calendar files
  - Calendar export

#### Link Management
- **`supabase/functions/reprocess-links/index.ts`**
  - Reprocess extracted links
  - Link classification update

- **`supabase/functions/translate-link-title/index.ts`**
  - Translate link titles
  - Multilingual support

#### HTML Parsing Service
- **`supabase/functions/html-parser-service/index.ts`**
  - HTML parsing service
  - Text extraction from HTML
  - Python service integration

- **`python-html-parser/app.py`**
  - Python HTML parser service
  - BeautifulSoup-based parsing

#### Newsletter Processing
- **`supabase/functions/process-newsletters/index.ts`**
  - Newsletter-specific processing
  - Newsletter content extraction

#### Contact Management
- **`supabase/functions/reprocess-contacts/index.ts`**
  - Reprocess contacts
  - Contact vetting update

#### Frontend Components
- **`src/components/OksoAssistant.tsx`**
  - AI assistant UI component
  - Chat interface

- **`src/components/EmailProcessor.tsx`**
  - Email processing UI
  - Email integration

- **`src/components/KnowledgeBaseUpload.tsx`**
  - Knowledge base upload UI
  - Document upload

- **`src/components/Dashboard.tsx`**
  - Main dashboard component
  - Overview and navigation

### Main Entry Points
- **Frontend**: `src/main.tsx` (React application)
- **Edge Functions**: `supabase/functions/*/index.ts` (Deno edge functions)
- **Python Service**: `python-html-parser/app.py` (Flask/FastAPI service)

### Documentation
- **`docs/README.md`**: Documentation index
- **`docs/data-workflow.md`**: Complete data workflow documentation
- **`docs/email-processing.md`**: Email processing pipeline
- **`docs/knowledge-base.md`**: Knowledge base documentation
- **`docs/ai-assistant.md`**: AI assistant documentation
- **`docs/privacy-security.md`**: Privacy and security documentation

---

## Common Patterns Across Repositories

### Synthetic Population Generation
- **VoxPop-SynC-GUI**: Generates synthetic populations from census data (SynC methodology)
- **VoxPopAI-KB**: Indexes and manages synthetic populations
- **voxpop-crew**: Uses synthetic populations for simulation

### Vector Embeddings
- **VoxPopAI-KB**: SentenceTransformers (local)
- **doctor-okso**: OpenAI embeddings (cloud)

### Database
- **VoxPop-SynC-GUI**: CSV/Parquet output files
- **VoxPopAI-KB**: PostgreSQL with pgvector
- **doctor-okso**: Supabase (PostgreSQL with pgvector)

### API Framework
- **VoxPop-SynC-GUI**: Command-line interface (Python)
- **VoxPopAI-KB**: FastAPI (Python)
- **voxpop-crew**: FastAPI (Python)
- **doctor-okso**: Supabase Edge Functions (TypeScript/Deno)

### LLM Integration
- **VoxPopAI-KB**: Multi-agent orchestration
- **voxpop-crew**: CrewAI framework
- **doctor-okso**: OpenAI GPT-4 with function calling

### Data Flow
1. **VoxPop-SynC-GUI**: Generates synthetic populations from census data
2. **VoxPopAI-KB**: Indexes and manages synthetic populations for querying
3. **voxpop-crew**: Uses synthetic populations for simulation and visualization
4. **doctor-okso**: Independent privacy-preserving knowledge base system

---

## Notes

- All code is publicly accessible for review

