# Consulting Reports Q&A

## Solution

### Overview
This solution implements a basic RAG (Retrieval Augmented Generation) system for processing and querying consulting reports with strict data isolation between tenants. The current implementation focuses on core functionality and provides a foundation for further enhancements.

#### Document Parsing Approach
- **Hierarchy Preservation:** Accurate extraction of document hierarchy (sections, subsections, tables) is critical for consulting reports.
- **Current Limitation:** Basic parsing modes (Unstructured, LlamaParse default) do not reliably preserve hierarchy; outputs are often flat or loosely structured.
- **Ongoing Evaluation:** We are actively experimenting with advanced LlamaParse modes (LVM, agent-based parsing) to improve structure extraction. Outputs are compared for section fidelity and ease of downstream use.
- **Next Steps:** Update the pipeline to use the parsing mode that best preserves hierarchy and document order.

### Architecture
- **Document Processing**: Unstructured + LlamaIndex for PDF parsing
- **Vector Storage**: Weaviate with multi-tenancy
- **Query Interface**: Streamlit UI with analytics dashboard
- **Security**: RBAC with tenant isolation
- **Validation**: Response validation and hallucination detection

### Key Features
- Multi-tenant vector storage with strict data isolation
- Hierarchical document processing preserving document structure
- Real-time query analytics and performance monitoring
- Response validation and quality checks
- Interactive UI with tenant-specific access

### Technical Stack
- **Vector Database**: Weaviate Cloud (multi-tenant enabled)
- **Embedding Model**: OpenAI text-embedding-3-small
- **Document Processing**: Unstructured, LlamaIndex
- **Parallel Processing**: Ray for distributed document processing
- **Frontend**: Streamlit
- **Analytics**: Custom query tracking with Plotly visualizations

### Project Setup (Local Development)
1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/#uv) and Python 3.12 if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```
3. Set up your `.env` file in the project root with all required API keys and config (see example in this README).

---

### Running the Application with Docker

**1. Build the Docker image:**
```bash
docker build -t sherpa-app .
```

**2. Run the Docker container:**
```bash
docker run -p 8000:8000 -p 8501-8510:8501-8510 sherpa-app
```
- FastAPI backend will be available at: http://localhost:8000
- Streamlit UI will be available at the port printed in the terminal (commonly http://localhost:8501)
- The Streamlit port is dynamic; check the logs for the correct URL.

**3. (Optional) Run the document processing pipeline on startup:**
```bash
docker run -e RUN_PIPELINE_ON_START=true -p 8000:8000 -p 8501-8510:8501-8510 sherpa-app
```

**To save output files (figures, processed_outputs) to your local machine:**
```bash
docker run -e RUN_PIPELINE_ON_START=true \
  -v $(pwd)/figures:/app/figures \
  -v $(pwd)/processed_outputs:/app/processed_outputs \
  -p 8000:8000 -p 8501-8510:8501-8510 sherpa-app
```
This will write all pipeline outputs to `figures/` and `processed_outputs/` in your current directory.

---

### Running the TruLens Dashboard (outside Docker)

The TruLens dashboard is **not** started by the Docker container. To run it locally:

1. **Activate your virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Install dependencies (if not already done):**
   ```bash
   uv sync
   ```
3. **Run the dashboard:**
   ```bash
   python src/scripts/run_trulens_dashboard.py
   ```
4. **Check the terminal for the dashboard URL** (e.g., `http://localhost:55204`) and open it in your browser.

---

### Environment Variables Example
Create a `.env` file in the project root:
```env
# Weaviate configuration
WEAVIATE_URL=your-weaviate-url
WEAVIATE_API_KEY=your-weaviate-api-key

# Tenant-specific passwords
WEAVIATE_BAIN_ADMIN_PASSWORD=your-bain-password
WEAVIATE_BCG_ADMIN_PASSWORD=your-bcg-password
WEAVIATE_MCK_ADMIN_PASSWORD=your-mck-password

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY=your-azure-openai-key
ENDPOINT_URL=your-azure-openai-endpoint
EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-3-small
DEPLOYMENT_NAME=your-deployment-name

# Other API Keys
LLAMA_API_KEY=your-llama-api-key
```
*Never commit the actual API keys to version control.*

---

### Notes
- The Docker image only starts the FastAPI backend and Streamlit UI.
- The TruLens dashboard must be run separately on your host machine.
- After starting the Docker container, always check the terminal for the correct Streamlit URL.
- For development or debugging, you can still use `entrypoint.sh` directly (after `chmod +x entrypoint.sh`).

---

### Quick Start

After completing the setup above:
1. Ensure your `.env` file is configured with the required API keys and endpoints.
2. Start the application:
   ```bash
   ./entrypoint.sh
   ```
   - FastAPI backend: http://localhost:8000
   - Streamlit UI: http://localhost:8501

3. (Optional) Run the document pipeline on startup:
   ```bash
   RUN_PIPELINE_ON_START=true ./entrypoint.sh
   ```

---

### Scripts & Performance Profiling

- **entrypoint.sh**: Main script to set up environment and launch all services.
- **store_vectors.py**: Script to process and store documents in Weaviate.
- **Scalene**: Installed for performance profiling. To profile any script, run:
  ```bash
  scalene src/your_script.py
  ```
  Replace `src/your_script.py` with the script you want to profile (e.g., `src/store_vectors.py`).

---

### Document Parsing: llama-parse & unstructured

- **llama-parse**: Used for extracting structured data (text, hierarchy, metadata) from PDFs and complex documents. It returns a list of Document objects, each with attributes like `text` and `metadata`. The pipeline iterates over these objects to extract content and hierarchy information.
- **unstructured**: Used as a fallback or for additional parsing (e.g., extracting tables, images, or diagrams not handled by llama-parse). It helps ensure all valuable data is extracted, even from non-linear or diagram-heavy slides.
- The codebase is designed to handle both tools seamlessly, maintaining document structure and hierarchy for downstream processing.

---

### Additional Notes
- **Security**: Do not commit any API keys or secrets. Use the `.env` file for all sensitive configuration.
- **Scalability**: The app is designed for easy extension and scaling. For high-volume or production use, consider deploying with a process manager or container orchestration.
- **Testing & Validation**: Output validation and prompt testing are included to minimize hallucinations and ensure robust model responses.

---

For issues or contributions, please open an issue or pull request.

1. Start the backend services:
```bash
# From the project root
# Start the document processing service
python src/main.py

# Start the FastAPI backend (in a new terminal)
PYTHONPATH=src uvicorn src.api.main:app --reload --port 8000
```

2. Launch the Streamlit UI (in a new terminal):
```bash
# From the project root
streamlit run src/ui/app.py
```

The application will be available at:
- FastAPI backend: http://localhost:8000
- Streamlit UI: http://localhost:8501

### Test Data
The repository includes a `test_data` directory with sample PDFs from each tenant for quick testing and evaluation. The pipeline will automatically process all PDFs found in `mbb_ai_reports` —no manual setup is required.

### Multi-tenant Support
- The system supports multiple tenants (Bain, BCG, McKinsey) with strict data isolation
- Each tenant's documents are stored and queried separately
- ⚠️ Authentication layer needs to be implemented for secure tenant access
- Current implementation focuses on demonstrating multi-tenant querying capabilities

### Current Limitations and Future Enhancements

- **Performance Optimization** (Planned)
  - Batch processing for very large document sets
  - Improved caching for frequently accessed content
  - Further query optimization for low-latency responses

- **Security** (Planned)
  - Full authentication system for tenant access (current: multi-tenant isolation, RBAC, API key validation)
  - Enhanced audit logging and request tracing

- **Known Issues**
  - ⚠️ Section IDs are currently identical across different documents, which may affect document relationship tracking
  - Planned: Generate unique section IDs using document identifiers for robust hierarchy

- **Scalability**
  - System is designed for multi-tenant scaling and can be extended with connection pooling and distributed processing
  - Query result caching and horizontal scaling are planned for production deployments

### Directory Structure
```
src/
├── processors/         # Document processing logic
├── utils/              # Shared utilities (weaviate_setup.py, auth.py, validation.py)
├── ui/                 # Streamlit interface
├── store_vectors.py    # Script to process and store documents
└── main.py             # Main processing script
```

### Challenges & Solutions

#### Multi-tenant Data Isolation
- **Implemented**: Weaviate multi-tenancy with RBAC to ensure strict data separation between consulting firms.

#### Document Processing
- **Enhanced parsing** for complex, non-linear consulting reports using llama-parse and unstructured.
- **Preserved document hierarchy** and improved error handling for malformed PDFs.
- **Image extraction** with coordinate mapping for diagrams and charts.
- **Planned**: Unique section IDs (incorporating document identifiers) and improved relationship tracking.

#### Vector Storage
- **Improved persistence**: Weaviate collections now support incremental, tenant-specific additions without overwriting existing vectors.


#### 4. Response Quality
**Challenge**: Ensuring high-quality responses without hallucinations.
**Solution**: Implemented validation layer with:
- Confidence score thresholds
- Hallucination detection
- Source validation
- Metadata verification

### Performance Monitoring
The system includes:
- Query latency tracking
- Confidence score monitoring
- Usage analytics by tenant
- Response validation metrics

### Security Considerations
1. Tenant isolation through Weaviate multi-tenancy
2. Role-based access control
3. API key management
4. Request validation
5. Data access logging

## 📌 Objective
You've been provided with a set of reports, articles and presentations on the topic of artificial intelligence, sourced from top-tier consulting firms (McKinsey, Bain and BCG). The documents include a mix of data — text, tables, charts, and diagrams. Your task is to build a prototype application, within 48 hours, that allows users to explore, synthesise and interrogate the content of these reports using AI-powered techniques.

- Python is preferred, but we welcome other languages if they're your strength. Feel free to use any packages you wish
- Likewise, we can provide you with AzureOpenAI credentials, but you are also welcome to use an LLM of your choice.

### Deliverables

#### Must-Haves
- A working codebase / repository that offers the user some ability to interact with the data. This can be as simple as the command line if you want to focus more heavily on the backend logic, but it could also be a fancy, deployed, UI if you want to show off your end-to-end development skills.
    - Please invite the following user to your repository: https://github.com/OLT2000
- README file documenting your thought process and set up instructions

#### Nice-to-Haves
1. **Complex Data Parsing**
    1. Are you able to utilise all of the data contained in the reports (i.e. including the tables, charts and images which contain valuable information)? ⚠️ (Partially Implemented)
     - ✅ Extracted images with precise coordinates from PDFs
     - ✅ Stored image metadata including position, size, and context
     - ⏳ CLIP model integration for image vectorization and semantic search
     - ⏳ Image-text cross-modal retrieval for relevant visuals during queries
     - ⏳ Table extraction and structured data parsing
    2. Typical consulting powerpoint decks contain content which does not follow a linear, logical flow like a Word document. Diagrams such as flow-charts use structure to add hierarchy to text. ✅ (Implemented hierarchical document processing with section paths)
2. **Authentication / RBAC** ✅
    1. You may want to consider simulating user authentication, data siloes or user permissioning / access rights. ✅ (Implemented RBAC with user roles)
    2. Consulting data is often extremely confidential. Accidentally leaking data between companies, or even internal teams, could be detrimental to us — security must be paramount. ✅ (Implemented strict multi-tenant isolation in Weaviate)
3. **Scalable Output Testing**
    1. What frameworks did you use to evaluate the quality of your model choices or prompts? ✅ (Implemented confidence score tracking and analytics)
    2. As we explore different solutions to our data problems, we need a way to reliably compare model performance. ✅ (Added performance monitoring dashboard)
4. **Model output validation** ✅
    1. How are you checking for hallucinations and response structures? ✅ (Implemented validation layer with hallucination detection)
5. **Agentic Integration with APIs**
    1. Users may want to perform common desk research about an industry or company that may not be available in the existing datasets ⏳ (Planned for future)
6. **Scalability** ⚠️
    1. Have you addressed the problem of scalability? ✅ (Implemented batch processing and tenant isolation)
    2. Is your application set up to handle large volumes of concurrent requests? How are you ensuring systems experience no downtime ⏳ (Connection pooling and caching planned)

Legend:
- ✅ Implemented
- ⚠️ Partially Implemented
- ⏳ Work in Progress/Planned

### Scalability Implementation ⚠️

- **Parallel & Batch Processing:**
  - Ray is used for parallel document processing, distributing work across available CPU cores.
  - Batch processing is implemented for efficient handling of large document sets and tenants.
  - This approach significantly reduces processing time for bulk PDF ingestion and vectorization.

- **Infrastructure:**
  - ✅ Tenant isolation and batch processing are implemented.
  - ⏳ Planned: Connection pooling, advanced caching, and load balancing for distributed query handling.

**Why Ray?**
- Native Python integration and simple API
- Efficient memory management for large documents
- Built-in fault tolerance
- Scalable from laptop to cluster

*Note: Ray is currently used for local and single-node parallelism. Cluster deployment and advanced distributed features are planned for future releases.*

---
### Author
**Zahara Miriam**  
Email: miriam_z@icloud.com