# Work Sample Technical Overview

This document provides a technical overview of data science, machine learning, and engineering work across four VoxPopAI repositories. It explains the methodologies, architectures, and implementations used for synthetic population generation, vector embeddings, LLM agent systems, and privacy-preserving data processing.

## Executive Summary

This work sample includes implementations of:

1. **Synthetic Population Generation Pipeline**: Implementing SynC methodology (Gaussian Copula) with neural networks to generate 5.2M+ synthetic personas from census data
2. **Large-Scale Data Management**: Creating and managing synthetic populations with 44 attributes per persona
3. **Vector Embeddings & Semantic Search**: Implementing multi-strategy retrieval systems using pgvector and OpenAI embeddings
4. **LLM Agent Orchestration**: Building sophisticated multi-agent systems with planning, reasoning, retrieval, and critique capabilities
5. **Privacy-Preserving Data Processing**: Implementing comprehensive PII redaction and privacy gates
6. **API Design & Integration**: Creating FastAPI endpoints that expose complex data science functionality
7. **Geographic Visualization**: Mapping and visualizing synthetic populations across Australian Statistical Areas

All implementations include privacy-preserving techniques, governance mechanisms, and responsible data handling practices.

---

## Repository 1: VoxPop-SynC-GUI

### Overview

VoxPop-SynC-GUI is the foundational synthetic population generation pipeline that creates synthetic personas from aggregated census data using the SynC (Synthetic data generation via Gaussian Copula) methodology. It uses statistical modeling, neural networks, and large-scale data generation techniques.

**Academic Foundation**: This implementation is based on the SynC framework described in Li, Zhao & Fu (2020) *"SynC: A Copula based Framework for Generating Synthetic Data from Aggregated Sources"* (arXiv:2009.09471). The paper PDF (`2009.09471v1.pdf`) is included in the repository root for reference.

### Key Data Science Components

#### 1. SynC Pipeline Implementation

**Files**:
- `models/synthetic_populations.py` - Main pipeline orchestration
- `models/functions.py` - Core statistical and ML functions
- `main.py` - Command-line interface and configuration management

**Purpose**: Generates synthetic populations that maintain statistical properties of real census data while creating individual-level profiles.

**SynC Methodology** (based on Li et al. 2020):

The pipeline implements a four-phase approach:

**Phase 0: Pre-processing**
- Input cleaning and formatting
- One-hot encoding for categorical variables (`to_dummy()`)
- Data normalization and scaling

**Phase 1: Dependency Modeling (Gaussian Copula)**

**What it is**: A Gaussian Copula is a statistical technique that models the dependency structure between multiple variables by separating their correlation structure from their individual marginal distributions. It uses a multivariate normal distribution to capture correlations, then transforms the results to match the desired marginal distributions.

**Why it's used here**: Census data is aggregated (we only know totals, not individual records), but we need to generate realistic synthetic individuals. The Gaussian Copula allows us to:
- Preserve the correlation patterns observed in aggregated census data
- Generate synthetic individuals whose joint characteristics match real-world dependencies (e.g., age and income are correlated)
- Create realistic combinations of attributes that would be impossible with independent sampling

The `sample_data()` function implements Gaussian Copula sampling to model feature dependencies:

```python
def sample_data(df, n_samples=None, verbose=False):
    """
    Generates synthetic samples from the empirical correlation structure 
    and marginal distributions using Gaussian Copula.
    """
    # Calculate correlation matrix from aggregated data
    corr_matrix = df.corr().values
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Force symmetry
    
    # Ensure positive definiteness for multivariate normal sampling
    def nearest_positive_definite(mat):
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals_clipped = np.clip(eigvals, 1e-4, None)
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    # Generate multivariate normal samples (Gaussian Copula)
    m = df.shape[1]
    x = np.random.multivariate_normal(
        mean=np.zeros(m), 
        cov=corr_matrix, 
        size=n_samples
    )
    u = norm.cdf(x)  # Transform to uniform [0,1] via normal CDF
    
    # Apply inverse CDF for marginal distributions
    synthetic_data = np.zeros_like(u)
    for i in range(m):
        col_data = df.iloc[:, i].values
        value_range = col_data.max() - col_data.min()
        
        if value_range < 1:
            # Beta distribution for proportions
            mean = np.mean(col_data)
            var = np.var(col_data)
            alpha = ((1 - mean) / var - 1 / mean) * mean ** 2
            beta_param = alpha * (1 / mean - 1)
            synthetic_data[:, i] = beta.ppf(u[:, i], alpha, beta_param)
        else:
            # Log-normal distribution for continuous values
            mean = np.mean(np.log(col_data[col_data > 0]))
            sigma = np.std(np.log(col_data[col_data > 0]))
            synthetic_data[:, i] = lognorm.ppf(u[:, i], s=sigma, scale=np.exp(mean))
    
    return synthetic_data, df.columns.tolist()
```

**Phase 2: Predictive Model Fitting**

**What it is**: Neural networks are machine learning models that learn complex non-linear relationships between input features (core demographics like age and gender) and output features (other attributes like income, education, ethnicity). They use multiple layers of interconnected nodes to capture patterns that simpler models cannot.

**Why it's used here**: After generating core demographics (age × gender), we need to assign other attributes to each synthetic individual. Neural networks are ideal because:
- They learn how demographic characteristics predict other attributes (e.g., how age and gender relate to income levels)
- They capture non-linear relationships and interactions between variables
- They generate probability distributions for each attribute, allowing realistic variation rather than deterministic assignments
- They can handle high-dimensional categorical outputs (many income brackets, education levels, etc.)

The `model_to_pop()` function trains neural networks to predict non-core features from core demographics:

```python
def model_to_pop(core, target, input_data, verbose=False, max_retries=3):
    """
    Trains neural network to predict feature distributions from core demographics.
    """
    # Split data: 70% training, 30% validation
    ind = np.random.choice(core.shape[0], size=int(0.3 * core.shape[0]), replace=False)
    train_x = core.drop(core.index[ind]).values
    train_y = target.drop(target.index[ind]).values
    val_x = core.iloc[ind].values
    val_y = target.iloc[ind].values
    
    # Build neural network architecture
    if np.all(train_y <= 1):
        # Classification head for categorical features
        inputs = layers.Input(shape=(train_x.shape[1],))
        x = layers.Dense(
            max(4, power_two(train_x.shape[1]) // 2), 
            activation='relu'
        )(inputs)
        x = layers.Dense(
            max(2, power_two(train_x.shape[1]) // 4), 
            activation='relu'
        )(x)
        outputs = layers.Dense(train_y.shape[1], activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['mae']
        )
    else:
        # Regression head for continuous features
        inputs = layers.Input(shape=(train_x.shape[1],))
        x = layers.Dense(
            max(4, power_two(train_x.shape[1]) // 2), 
            activation='relu'
        )(inputs)
        x = layers.Dense(
            max(2, power_two(train_x.shape[1]) // 4), 
            activation='relu'
        )(x)
        outputs = layers.Dense(train_y.shape[1])(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train model
    history = model.fit(
        train_x, train_y, 
        epochs=5, 
        batch_size=512, 
        validation_data=(val_x, val_y)
    )
    
    # Generate predictions for all individuals
    test_y = model.predict(input_data, batch_size=32)
    return pd.DataFrame(test_y, columns=target.columns)
```

**Phase 3: Marginal Scaling**

**What it is**: Marginal matching is an iterative optimization algorithm that adjusts synthetic individual assignments to ensure the aggregate totals exactly match known census marginals (totals for each category). It works by identifying discrepancies between synthetic counts and target counts, then reassigning individuals to eliminate these differences.

**Why it's used here**: While neural networks generate realistic probability distributions, they don't guarantee that the final counts match census totals exactly. Marginal matching is critical because:
- Census data provides exact totals for each category (e.g., exactly 1,250 people in income bracket $50k-$75k)
- Synthetic populations must match these totals to be statistically valid for analysis
- It ensures the synthetic data preserves the statistical properties of the original census data
- It maintains realism while enforcing exact constraints (implements Algorithm 3 from the SynC paper)

The `match_marginal()` function ensures synthetic data exactly matches census marginals:

```python
def match_marginal(output, marginals, varnames, probs_df, max_iter=1000, verbose=False):
    """
    Iterative algorithm to match synthetic assignments to target marginals.
    Implements Algorithm 3 from SynC paper.
    """
    n_individuals, n_categories = output.shape
    matched = np.zeros((n_individuals, n_categories))
    
    # Initial assignment: assign each individual to most probable category
    initial_assignments = np.argmax(output, axis=1)
    for i, cat in enumerate(initial_assignments):
        matched[i, cat] = 1
    
    empirical_marginals = matched.sum(axis=0)
    iteration = 0
    
    # Iterative matching loop
    while not np.all(empirical_marginals == marginals) and iteration < max_iter:
        iteration += 1
        diff = empirical_marginals - marginals
        
        # Identify over-represented categories
        over_indexed = np.where(diff > 0)[0]
        # Identify under-represented categories
        under_indexed = np.where(diff < 0)[0]
        
        # Un-assign weakest predictions in over-represented categories
        for cat in over_indexed:
            assigned_indices = np.where(matched[:, cat] == 1)[0]
            probs_for_assigned = output[assigned_indices, cat]
            remove_count = int(min(diff[cat], len(assigned_indices)))
            remove_indices = assigned_indices[np.argsort(probs_for_assigned)[:remove_count]]
            matched[remove_indices, cat] = 0
        
        # Re-assign to under-represented categories proportionally
        for cat in under_indexed:
            unassigned_individuals = np.where(matched.sum(axis=1) == 0)[0]
            if len(unassigned_individuals) > 0:
                probs_for_unassigned = output[unassigned_individuals, cat]
                add_count = int(min(-diff[cat], len(unassigned_individuals)))
                add_indices = unassigned_individuals[
                    np.argsort(probs_for_unassigned)[-add_count:]
                ]
                matched[add_indices, cat] = 1
        
        empirical_marginals = matched.sum(axis=0)
    
    # Convert to category names
    assigned_categories = []
    for i in range(n_individuals):
        assigned_cat = np.where(matched[i, :] == 1)[0]
        assigned_categories.append(varnames[assigned_cat[0]] if len(assigned_cat) > 0 else "")
    
    return assigned_categories
```

**Data Science Techniques**:
- Gaussian Copula modeling for dependency structure
- Neural network training for feature prediction
- Iterative marginal matching algorithms
- Statistical sampling and distribution fitting

#### 2. Configuration-Driven Pipeline

**File**: `main.py`, `config.yaml`

**Purpose**: Fully configurable pipeline controlled by YAML configuration files.

**What it is**: Configuration-driven design separates data and processing logic from code, allowing all parameters, file paths, and feature specifications to be defined in external YAML files rather than hardcoded in Python.

**Why it's used here**: This approach provides several critical benefits for synthetic population generation:
- **Flexibility**: Add new demographic features or data sources without modifying code
- **Reproducibility**: Exact configuration can be saved and reused to regenerate identical populations
- **Scalability**: Process different geographic regions or datasets by simply changing configuration files
- **Maintainability**: Non-programmers can modify data sources and parameters without touching code
- **Experimentation**: Easy to test different feature combinations or processing parameters

**Configuration Structure**:
- **Project metadata**: Project ID and name
- **Data sources**: File paths for core data and feature batches
- **Geographic filtering**: Location column and target postal codes
- **Feature specifications**: Column indices and names for each feature type
- **Processing parameters**: Random seeds, verbose mode, reference ages
- **Output settings**: Directory structure and file naming

**Example Configuration** (`config.yaml`):
```yaml
project:
  id: voxpop_sync_census
  name: VoxPop SynC Census Data

files:
  core_data: data/supermarket/sync_ready/mock_core.csv
  output_dir: output

geography:
  column: location
  target_postcode: Acacia Gardens

core_features:
  columns: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  names: ["age_15-19 years", "age_20-24 years", ..., "gender_Male"]

feature_batches:
  - name: income
    source_file: data/supermarket/sync_ready/mock_income.csv
    columns: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    names: ["income_$1,250-$1,499", "income_$1,500-$1,749", ...]

processing:
  ref_age: 18
  ref_col: 1
  seed: 123
  verbose: true
```

**Pipeline Execution** (`main.py`):
```python
def load_config(config_path):
    """Load and validate YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate required sections
    required_sections = ['files', 'geography', 'core_features', 'feature_batches']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    return _resolve_relative_paths(config, config_path)

# Main pipeline execution
config = load_config('config.yaml')
run_synthetic_populations(config, verbose=True)
```

**Benefits**:
- No code changes needed to add new features
- Reproducible runs with seed control
- Flexible geographic filtering
- Easy experimentation with different configurations

**Data Science Techniques**:
- Configuration management
- Parameter validation
- Path resolution and file handling

#### 3. Feature Generation

**File**: `models/functions.py`

**Purpose**: Generates 39+ attributes per synthetic individual across multiple categories.

**What it is**: Feature generation is the process of assigning multiple demographic, behavioral, and personality attributes to each synthetic individual. This involves processing different data sources (income, education, ethnicity, etc.) sequentially, using the SynC pipeline to ensure realistic attribute combinations.

**Why it's used here**: Synthetic populations need rich individual profiles to be useful for simulation and analysis. The feature generation process:
- **Sequential Processing**: Each feature category (income, education, etc.) is processed independently, building up complete profiles incrementally
- **Dependency Preservation**: Attributes are generated in order, with later attributes depending on earlier ones (e.g., income depends on age and gender)
- **Realistic Combinations**: Ensures attribute combinations match real-world patterns (e.g., high education correlates with higher income)
- **Comprehensive Coverage**: Generates 39+ attributes covering demographics, behaviors, and personality traits for each individual

**Core Processing Function** (`process_file()`):
```python
def process_file(file_path, var_name, postal, index, category_names, 
                core_file_path='data/core.csv', 
                individual_file_path='data/individual.csv', 
                ref_age=18, ref_col=1, verbose=False):
    """
    Process a demographic file and generate synthetic data using neural networks.
    This is the core of the synthetic population generation pipeline.
    """
    # Load raw demographic data and core features
    raw = pd.read_csv(file_path, index_col=0)
    core = pd.read_csv(core_file_path)
    individual = pd.read_csv(individual_file_path)
    
    # Normalize target variables by reference column
    if ref_col == 0:
        target = raw.iloc[:, index]
    else:
        target = raw.iloc[:, index].div(raw.iloc[:, ref_col], axis=0)
    
    # Prepare input data from individual demographics (one-hot encoding)
    input_data = to_dummy(individual['Demographics'], categories=core.columns.tolist())
    
    # Train neural network to predict feature distributions
    try:
        output = model_to_pop(core, target, input_data.values, verbose=verbose)
    except Exception as e:
        # Fallback to uniform distribution if neural network fails
        n_samples = input_data.shape[0]
        n_categories = len(target.columns)
        fallback_probs = np.ones((n_samples, n_categories)) / n_categories
        output = pd.DataFrame(fallback_probs, columns=target.columns)
    
    return output
```

**Feature Categories**:

**Basic Demographics (3)**:
- Age bracket (e.g., "PP_M30_34")
- Numeric age value
- Gender (M/F)

**Census Features (7)**:
- Unpaid childcare, need assistance, language at home
- Unpaid assistance, unpaid work
- Indigenous language, health status

**Core Demographics (9)**:
- Marital status (registered and social)
- Parental status and child type
- Spouse/partner details
- Number of children
- Income bracket
- Citizenship, religion

**Persona-Based Supermarket Features (6)**:
Generated using realistic behavioral logic:
- ShoppingFrequency ← age + household_size
- BasketSize ← ShoppingFrequency + household_size
- PriceSensitivity ← age + income
- PreferredChannel ← age
- PrivateLabelPreference ← age
- CategorySpendBias ← household_composition

**Big 5 Personality Traits (15)**:
- Levels: Low/Neutral/High for each trait
- Scores: 0-100 scale
- Descriptions: Human-readable trait descriptions
- Traits: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism

**Data Science Techniques**:
- Conditional feature generation
- Behavioral modeling
- Personality trait assignment
- Feature dependency modeling

#### 4. All-Locations Processing

**File**: `models/synthetic_populations.py` - `process_all_locations()`

**Purpose**: Processes multiple geographic locations with robust error handling and resume capability.

**Capabilities**:
- **Error Handling**: Graceful failures - if one location fails, processing continues
- **Progress Tracking**: JSON progress file tracks completed, failed, and skipped locations
- **Resume Capability**: Can restart and skip already completed locations
- **Comprehensive Logging**: Detailed logs with timestamps, statistics, and error details
- **Individual Files**: Each location saved separately for verification
- **Combined Output**: Final merged dataset with all successful locations

**Data Science Techniques**:
- Batch processing
- Error recovery
- State management
- Progress tracking

#### 5. Output Generation

**File**: `models/synthetic_populations.py`

**Purpose**: Generates output files in multiple formats for downstream use.

**Output Files**:
- `core.csv`: Core demographic probabilities
- `individual.csv`: Individual-level synthetic data
- `synthetic_population_<location>.csv`: Final synthetic population per location
- `all_locations_synthetic_population_<timestamp>.csv`: Combined results

**Output Structure**:
- Partitioned by geographic hierarchy (state, SA4, SA3, SA2, SA1)
- Ready for ingestion into VoxPopAI-KB pipeline
- Compatible with Parquet format conversion

**Data Science Techniques**:
- Data serialization
- File format conversion
- Partitioning strategies
- Timestamp-based versioning

### Privacy & Governance

- Data generated from public ABS census data
- Statistical properties preserved without individual identification
- Geographic aggregation protects privacy

---

## Repository 2: VoxPopAI-KB

### Overview

VoxPopAI-KB is a knowledge base service that manages synthetic personas, provides vector-based retrieval, and orchestrates LLM-enhanced agents. It uses database design, vector embeddings, and agent systems.

### Key Data Science Components

#### 1. Synthetic Persona Index Population

**File**: `kb_service/personas/populate_index.py`

**Purpose**: Populates a PostgreSQL database index from Parquet files containing synthetic persona data.

**Methodology**:
- Discovers Parquet files in a partitioned directory structure (by ABS state, SA4, SA3, SA2, SA1)
- Processes 5.2M+ personas across 415+ Parquet files
- Maps 44 persona attributes to database columns including:
  - Demographics (age, gender, income, household composition)
  - Cultural attributes (citizenship, language, religion)
  - Health and assistance needs
  - Personality traits (Big Five: agreeableness, conscientiousness, extraversion, neuroticism, openness)
  - Shopping behavior patterns
- Implements incremental updates using file hash comparison
- Generates deterministic UUIDs for persona identification

**Data Science Techniques**:
- Efficient batch processing of large datasets
- Incremental update strategies for scalability
- Database indexing for fast queries
- Data transformation and normalization

#### 2. Vector Embeddings & Retrieval

**Files**: 
- `kb_service/ingest/embeddings_local.py` - Embedding generation
- `kb_service/agents/retriever.py` - Advanced retrieval system

**Purpose**: Implements semantic search using vector embeddings stored in PostgreSQL with pgvector extension.

**Methodology**:
- Uses SentenceTransformers for local embedding generation (configurable model)
- Stores embeddings in pgvector-enabled PostgreSQL database
- Implements multiple retrieval strategies:
  - **Vector Search**: Cosine similarity search using embeddings
  - **Keyword Search**: Full-text search using PostgreSQL FTS
  - **Hybrid Search**: Combines vector and keyword results
- Multi-stage retrieval pipeline:
  1. Initial candidate retrieval (3x desired results)
  2. LLM-based re-ranking (optional)
  3. Metadata enrichment and final selection

**Data Science Techniques**:
- Embedding normalization for consistent similarity calculations
- Batch processing for efficient embedding generation
- Hybrid retrieval combining semantic and keyword matching
- Re-ranking using LLM for improved relevance

#### 3. LLM-Enhanced Agent System

**Files**:
- `kb_service/agents/planner.py` - Intelligent planning
- `kb_service/agents/reasoner.py` - Context-aware reasoning
- `kb_service/agents/retriever.py` - Advanced retrieval
- `kb_service/agents/critic.py` - Quality critique
- `kb_service/agents/distiller.py` - Response distillation
- `kb_service/agents/runner.py` - Agent orchestration

**Purpose**: Orchestrates multiple LLM agents to answer questions using persona profiles and retrieved knowledge.

**Architecture**:
1. **Planner**: Generates intelligent planning decisions using LLM
   - Analyzes question and persona context
   - Generates dynamic filters for retrieval
   - Selects exemplars for guidance
   - Determines optimal chunk limits

2. **Retriever**: Executes multi-strategy retrieval
   - Vector similarity search
   - Keyword-based search
   - Hybrid approaches
   - Freshness filtering

3. **Reasoner**: Generates context-aware answers
   - Builds comprehensive context from persona profile, chunks, and exemplars
   - Uses LLM to generate answers aligned with persona characteristics
   - Extracts citations and reasoning chains
   - Assesses answer confidence

4. **Critic**: Evaluates answer quality
   - Checks factual accuracy
   - Validates persona alignment
   - Assesses completeness
   - Provides improvement suggestions

5. **Distiller**: Refines final responses
   - Summarizes key points
   - Removes redundancy
   - Formats for presentation

**Data Science Techniques**:
- Multi-agent orchestration
- LLM prompt engineering
- Context assembly and management
- Quality assessment and validation

#### 4. Persona Hydration

**File**: `kb_service/hydrate/hydrate.py`

**Purpose**: Combines canonical persona data, regional knowledge packs, and memory into a unified persona profile.

**Methodology**:
- Reads canonical persona data from database (44 attributes)
- Joins regional knowledge packs (metrics, derived data, sources)
- Incorporates persona memory (traits, stances, interaction stats, notes)
- Assigns or retrieves display names
- Assesses persona relevance to questions

**Data Science Techniques**:
- Data joining and merging
- Profile enrichment
- Relevance scoring

#### 5. FastAPI Endpoints

**File**: `kb_service/api/main.py`

**Purpose**: Exposes data science functionality through RESTful API endpoints.

**Key Endpoints**:
- `POST /select-personas`: Select personas based on geo and demographic filters
- `POST /hydrate`: Hydrate persona profiles with canonical, regional, and memory data
- `POST /run-agent`: Execute LLM agent pipeline for question answering
- `POST /retrieve`: Advanced retrieval with multiple strategies
- `GET /persona/{persona_id}/memory`: Retrieve persona memory
- `GET /knowledge_pack/{region_code}`: Get regional knowledge pack

**Data Science Techniques**:
- API design for data science workflows
- Request/response validation
- Error handling and logging
- Authentication and authorization

### Privacy & Governance

- Database queries are scoped by region_code for data isolation
- Authentication required for all endpoints

---

## Repository 3: voxpop-crew

### Overview

voxpop-crew is a synthetic population simulation system that uses CrewAI agents to conduct interviews and surveys with synthetic personas. It includes geographic visualization capabilities using agent orchestration and data visualization techniques.

### Key Data Science Components

#### 1. Synthetic Population Management

**Files**:
- `crew_agents/voxpop_agents/personas.py` - Persona loading
- `voxpopai/backend/agents/persona_builder.py` - Persona construction
- `voxpopai/backend/routers/personas.py` - Persona API endpoints

**Purpose**: Manages synthetic personas for simulation and survey purposes.

**Methodology**:
- Loads personas from CSV/Parquet sources
- Supports filtering by geographic area (SA2 codes)
- Implements persona selection based on demographic criteria
- Integrates with VoxPopAI-KB for persona hydration

**Data Science Techniques**:
- Data loading and transformation
- Filtering and selection algorithms
- Integration with external services

#### 2. Geographic Visualization

**File**: `geo_tools/plot_synthetic.py`

**Purpose**: Creates visualizations of synthetic personas on geographic maps.

**Capabilities**:
- **Plot Selected Regions**: Visualize SA2 regions on maps
- **Plot Personas on Map**: Show persona distribution with geographic boundaries
- **Multi-Region Comparison**: Compare personas across multiple regions
- **Demographic Heatmaps**: Visualize demographic distributions across regions
- **Highlighted Personas**: Show regions with personas highlighted

**Data Science Techniques**:
- Geographic data processing using geopandas
- Spatial visualization
- Statistical aggregation by region
- Map generation and styling

**Visualization Functions**:
- `plot_selected_regions()`: Plot SA2 boundaries
- `plot_personas_on_map()`: Scatter plot of personas with boundaries
- `plot_multi_region_comparison()`: Side-by-side region comparisons
- `create_demographic_heatmap()`: Heatmap of demographic variables

#### 3. CrewAI-Based Simulation Agents

**Files**:
- `voxpopai/backend/agents/crewai_simulation_agent.py` - CrewAI simulation
- `crew_agents/voxpop_agents/crew.py` - CrewAI crew definition
- `crew_agents/voxpop_agents/enhanced_agents.py` - Enhanced agent implementations

**Purpose**: Uses CrewAI framework to orchestrate agents that simulate persona responses to questions.

**Architecture**:
- **Persona Agents**: Represent synthetic personas with their characteristics
- **Facilitator Agents**: Conduct interviews and guide conversations
- **Critic Agents**: Evaluate question quality and relevance
- **Coordinator**: Orchestrates multi-agent interactions

**Data Science Techniques**:
- Multi-agent systems
- Agent orchestration
- Response simulation
- Quality assessment

#### 4. Focus Group Facilitation

**Files**:
- `voxpopai/backend/agents/focus_group_facilitator/facilitator_agent.py`
- `voxpopai/backend/agents/focus_group_facilitator/interview_styles.py`
- `voxpopai/backend/agents/focus_group_facilitator/memory_manager.py`

**Purpose**: Conducts in-depth interviews with synthetic personas using different interview styles.

**Interview Styles**:
- **Exploratory**: Open-ended exploration of topics
- **Laddering**: Deep-dive into motivations and values
- **Narrative**: Story-based conversation

**Methodology**:
- Maintains conversation memory across turns
- Adapts questions based on persona responses
- Tracks context and themes
- Generates structured session outputs

**Data Science Techniques**:
- Conversation management
- Context tracking
- Adaptive questioning
- Session analysis

#### 5. Survey Composition & Analysis

**Files**:
- `voxpopai/backend/agents/survey_composer.py`
- `voxpopai/backend/agents/question_critic.py`
- `voxpopai/backend/routers/surveys.py`

**Purpose**: Composes surveys and analyzes responses from synthetic personas.

**Capabilities**:
- Survey question generation
- Question quality critique
- Response analysis and summarization
- Statistical analysis of responses

**Data Science Techniques**:
- Survey design
- Response analysis
- Statistical aggregation
- Theme extraction

#### 6. FastAPI Backend

**File**: `voxpopai/backend/app.py`

**Purpose**: Provides RESTful API for frontend and external integrations.

**Key Endpoints**:
- `/api/personas/simulate`: Run simulations with personas
- `/api/focus-groups/`: Manage focus group sessions
- `/api/surveys/`: Compose and analyze surveys
- `/api/question/critic`: Critique question quality
- `/api/geo/`: Geographic data endpoints

**Data Science Techniques**:
- API design for simulation workflows
- Streaming responses for long-running operations
- Error handling and logging

### Privacy & Governance

- Geographic data uses public ABS shapefiles

---

## Repository 4: doctor-okso

### Overview

doctor-okso is a privacy-preserving knowledge base system that processes school newsletters, extracts information, and provides an AI assistant. It uses PII redaction, vector embeddings, and multi-strategy semantic search.

### Key Data Science Components

#### 1. Email Processing Pipeline with Privacy Gate

**Files**:
- `supabase/functions/process-emails-v2/index.ts` - Email processing
- `supabase/functions/_shared/privacy-gate.ts` - PII redaction

**Purpose**: Processes emails (school newsletters) while automatically redacting personally identifiable information.

**Methodology**:

**Privacy Gate Process**:
1. **HTML Parsing**: Extracts text from HTML email content
2. **Section Filtering**: 
   - Drops sensitive sections (student of the week, birthdays, photos, awards)
   - Keeps useful sections (events, policies, announcements, schedules)
3. **PII Redaction**:
   - Emails → `[email]`
   - Phone numbers → `[phone]`
   - URLs → `[link]` (except approved domains)
   - Names (capitalized word sequences) → `[name]`
   - Student IDs → `[student-id]`
   - Social handles → `[social-handle]`
4. **Contact Extraction**: Extracts contacts through vetting gate
   - School emails → pending approval
   - Approved domains → pending approval
   - Personal emails → blocked
5. **AI Summarization**: Summarizes sanitized content
6. **Storage**: Stores processed content in database

**Data Science Techniques**:
- Text processing and parsing
- Pattern matching for PII detection
- Section classification
- Contact vetting algorithms
- AI-powered summarization

#### 2. Vector Embeddings & Knowledge Base

**Files**:
- `supabase/functions/process-knowledge-base-upload/index.ts`
- `supabase/functions/okso-assistant/index.ts` (semantic search)

**Purpose**: Creates a searchable knowledge base using vector embeddings.

**Methodology**:
- **Chunking**: Splits content into ~500 token chunks with 50 token overlap
- **Embedding Generation**: Uses OpenAI `text-embedding-3-small` model
- **Storage**: Stores chunks with embeddings in PostgreSQL using pgvector
- **Deduplication**: Uses SHA256 hashing to prevent duplicate content

**Data Science Techniques**:
- Text chunking strategies
- Vector embedding generation
- Similarity search using pgvector
- Content deduplication

#### 3. Multi-Strategy Semantic Search

**File**: `supabase/functions/okso-assistant/index.ts`

**Purpose**: Implements an AI assistant with multiple search strategies for answering questions.

**Search Strategies**:

1. **Semantic Search** (Vector Similarity):
   - Generates embedding for query
   - Searches knowledge base using cosine similarity
   - Best for conceptual questions

2. **Keyword Search** (Full-Text):
   - Uses PostgreSQL full-text search
   - Searches knowledge base and newsletter sections
   - Best for specific terms and exact phrases

3. **Date Search**:
   - Queries by date range
   - Searches calendar events, knowledge base, and newsletters
   - Best for time-specific queries

4. **Link Search** (Tag-Based):
   - Detects homework/link questions
   - Queries `extracted_links` table by tags
   - Returns classified links with metadata

**AI Agent Integration**:
- Uses OpenAI GPT-4 with function calling
- AI selects appropriate search strategies
- Combines results from multiple strategies
- Generates answers with source citations

**Data Science Techniques**:
- Multi-strategy retrieval
- AI-powered query routing
- Result combination and ranking
- Context assembly for LLM

#### 4. URL Classification System

**File**: `supabase/functions/process-knowledge-base-upload/index.ts`

**Purpose**: Extracts and classifies URLs from documents before privacy redaction.

**Methodology**:
- Extracts ALL URLs from raw content (before privacy gate)
- AI classifies each URL with:
  - Descriptive title
  - Description
  - Tags (homework, form, event, resource, etc.)
  - Link type (primary category)
  - Confidence score
- Stores in `extracted_links` table
- Deduplication by (user_id, url)

**Data Science Techniques**:
- URL extraction and parsing
- AI-powered classification
- Tag-based organization
- Deduplication strategies

#### 5. Contact Management & Vetting

**Files**:
- `supabase/functions/_shared/contacts.ts`
- `supabase/functions/_shared/privacy-gate.ts`

**Purpose**: Manages contact extraction with privacy vetting.

**Vetting Rules**:
- **School Emails**: Role-based (office@, admin@) + approved domain → pending
- **Phones/Fax**: All → pending for user review
- **URLs**: Approved domains (.edu.au, school domains) → pending
- **Personal Emails**: Blocked automatically

**Data Science Techniques**:
- Pattern matching for contact types
- Domain validation
- Approval workflow management

### Privacy & Governance

- Comprehensive PII redaction before processing
- Contact vetting with user approval
- User-scoped data with Row Level Security (RLS)
- All AI processing uses sanitized content

---

## Cross-Repository Patterns

### 1. Privacy and Governance Principles

All repositories include privacy-preserving techniques:

- **PII Redaction**: Automatic redaction of personally identifiable information
- **User Scoping**: Data isolation by user/region
- **Approval Workflows**: Vetting mechanisms for sensitive data

### 2. Vector Search Patterns

Both VoxPopAI-KB and doctor-okso implement vector embeddings:

- **Embedding Generation**: SentenceTransformers (KB) and OpenAI (okso)
- **Storage**: PostgreSQL with pgvector extension
- **Similarity Search**: Cosine similarity for semantic matching
- **Hybrid Approaches**: Combining vector and keyword search

### 3. LLM Integration Patterns

All repositories use LLMs strategically:

- **VoxPopAI-KB**: Multi-agent orchestration (planner, reasoner, critic)
- **voxpop-crew**: CrewAI agents for simulation and facilitation
- **doctor-okso**: AI assistant with function calling for search

**Common Patterns**:
- Context assembly from multiple sources
- Prompt engineering for specific tasks
- Function calling for tool use
- Response validation and critique

### 4. API Design Patterns

All repositories expose FastAPI endpoints:

- **RESTful Design**: Standard HTTP methods and status codes
- **Request/Response Validation**: Pydantic models for type safety
- **Error Handling**: Comprehensive error responses
- **Authentication**: Bearer token authentication
- **CORS**: Proper cross-origin resource sharing configuration

---

## Data Science Highlights

### Synthetic Population Generation Methodology

The synthetic population generation follows a two-stage process:

**Stage 1: SynC Pipeline (VoxPop-SynC-GUI)**
1. **Data Source**: Public Australian Bureau of Statistics (ABS) census data (aggregated)
2. **Generation Process**:
   - Gaussian Copula modeling for dependency structure
   - Neural network training for feature prediction
   - Marginal matching to ensure exact census totals
   - Behavioral feature generation using logical rules
   - Big 5 personality trait assignment
3. **Output**: Individual-level synthetic personas with 39+ attributes
4. **Scale**: Processes all locations across Australia

**Stage 2: Indexing & Management (VoxPopAI-KB)**
1. **Input**: Synthetic population CSV/Parquet files from SynC pipeline
2. **Processing**:
   - Converts to Parquet format with geographic partitioning
   - Populates PostgreSQL index with 44 attributes per persona
   - Enables fast queries and filtering
3. **Scale**: 5.2M+ synthetic personas across 415+ Parquet files
4. **Storage**: Partitioned Parquet files by geographic hierarchy (state, SA4, SA3, SA2, SA1)
5. **Indexing**: PostgreSQL database for fast queries

### Feature Engineering (44 Persona Attributes)

**Demographics**:
- Age band, gender, income band
- Household size, parent flag, child type
- Marital status (social and registered)

**Cultural**:
- Citizenship, language at home, English proficiency
- Religion, indigenous status, ancestry

**Health & Assistance**:
- Health status, need assistance flags
- Unpaid assistance, childcare, work

**Personality (Big Five)**:
- Agreeableness, conscientiousness, extraversion, neuroticism, openness
- Each with description, level, and score

**Shopping Behavior**:
- Basket size, category spend bias, preferred channel
- Price sensitivity, private label preference, shopping frequency

### Embedding Strategies

**VoxPopAI-KB**:
- Local embeddings using SentenceTransformers
- Configurable models (default: all-MiniLM-L6-v2)
- Batch processing for efficiency
- Fallback to dummy vectors if model unavailable

**doctor-okso**:
- OpenAI embeddings (text-embedding-3-small)
- Consistent embedding model for semantic search
- Chunk-based embedding generation

### Retrieval Strategies

**Vector Search**:
- Cosine similarity using normalized embeddings
- Configurable similarity thresholds
- Result ranking by similarity score

**Keyword Search**:
- PostgreSQL full-text search
- Keyword matching and phrase search
- Relevance ranking

**Hybrid Search**:
- Combines vector and keyword results
- Deduplication and re-ranking
- Balanced relevance scoring

**Multi-Stage Retrieval**:
- Initial candidate retrieval (3x desired results)
- LLM-based re-ranking (optional)
- Metadata enrichment
- Final selection

### Agent Orchestration

**VoxPopAI-KB Agent Pipeline**:
1. Planner → Generates filters and exemplars
2. Retriever → Executes multi-strategy retrieval
3. Reasoner → Generates context-aware answers
4. Critic → Evaluates answer quality
5. Distiller → Refines final response

**voxpop-crew Agent System**:
- CrewAI framework for agent orchestration
- Persona agents represent synthetic individuals
- Facilitator agents conduct interviews
- Critic agents evaluate questions
- Coordinator manages interactions

**doctor-okso AI Assistant**:
- Function calling for search tool selection
- Multi-strategy search execution
- Context assembly from results
- Answer generation with citations

---

## Diagnostic Plots and Visualizations

### Geographic Visualizations (voxpop-crew)

The `geo_tools/plot_synthetic.py` module provides several visualization functions:

1. **Selected Regions Plot**: Shows SA2 boundaries on maps
2. **Personas on Map**: Scatter plot of personas with geographic boundaries
3. **Multi-Region Comparison**: Side-by-side comparisons across regions
4. **Demographic Heatmaps**: Visualize demographic distributions across SA2 regions
5. **Highlighted Personas**: Show regions containing personas with highlighting

**Use Cases**:
- Validate synthetic population distribution
- Compare personas across geographic regions
- Analyze demographic patterns
- Visualize coverage of synthetic data

**Technical Details**:
- Uses geopandas for geographic data processing
- Matplotlib for visualization
- ABS shapefiles for SA2 boundaries
- Configurable styling and colors

---

## Summary

This work sample includes data science and engineering implementations across four interconnected repositories. The implementations include:

- **Synthetic Data Generation**: SynC methodology with Gaussian Copula and neural networks
- **Vector Embeddings**: Multiple embedding strategies using SentenceTransformers and OpenAI models
- **LLM Agent Systems**: Multi-agent orchestration with planning, reasoning, and critique capabilities
- **Privacy-Preserving Techniques**: PII redaction, privacy gates, and synthetic data generation
- **API Design**: FastAPI endpoints exposing data science functionality
- **Geographic Visualization**: Mapping and visualization of synthetic populations
- **Large-Scale Processing**: Handling of 5.2M+ synthetic personas across multiple repositories

All implementations include privacy-preserving techniques, data protection mechanisms, governance practices, and responsible data handling.

