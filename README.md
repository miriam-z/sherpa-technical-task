## Solution

### Overview
This solution implements a multi-tenant RAG (Retrieval Augmented Generation) system for processing and querying consulting reports with strict data isolation between tenants.

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

### Project Setup
1. Clone the repository

2. Install uv package manager:
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. Verify installation:
   ```bash
   uv version
   ```

4. Set up environment variables:
   ```bash
   WEAVIATE_URL=your-weaviate-url
   WEAVIATE_API_KEY=your-api-key
   OPENAI_API_KEY=your-openai-key
   ```

5. Create and activate virtual environment:
   ```bash
   # Create virtual environment with Python 3.12
   uv venv --python 3.12

   # Activate virtual environment
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

6. Install dependencies:
   ```bash
   # Sync dependencies from pyproject.toml
   uv sync
   ```

Note: uv is a fast, reliable Python package installer and resolver. For more information, visit [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#uv).

### Running the Project
1. Process documents:
```bash
python src/main.py
```
2. Launch UI:
```bash
streamlit run src/ui/app.py
```

### Directory Structure
```
src/
├── processors/      # Document processing logic
├── utils/          # Shared utilities
│   ├── weaviate_setup.py
│   ├── auth.py
│   └── validation.py
├── ui/             # Streamlit interface
└── main.py         # Main processing script
```

### Challenges & Solutions

#### 1. Multi-tenant Data Isolation
**Challenge**: Ensuring strict data separation between consulting firms.
**Solution**: Implemented Weaviate's multi-tenancy with proper RBAC configuration.

#### 2. Document Processing
**Challenge**: Handling complex consulting reports with non-linear structure.
**Solution**: 
- Enhanced document processor to handle None elements
- Added robust error handling for malformed PDFs
- Preserved document hierarchy and relationships
- Implemented image extraction with coordinate mapping
- Prepared for CLIP-based visual semantic search
- Built foundation for cross-modal retrieval (text-to-image, image-to-text)

#### 3. Vector Storage Persistence
**Challenge**: Initial implementation overwrote vectors when processing multiple tenants.
**Solution**: Modified Weaviate setup to preserve existing collections and add tenant-specific data incrementally.

#### 4. Response Quality
**Challenge**: Ensuring high-quality responses without hallucinations.
**Solution**: Implemented validation layer with:
- Confidence score thresholds
- Hallucination detection
- Source validation
- Metadata verification

### Work in Progress
1. **Complex Data Parsing**
- [ ] Table extraction and vectorization
- [ ] Image analysis and linking
- [ ] Flow-chart structure preservation

2. **Scalability**
- [ ] Connection pooling
- [ ] Query caching
- [ ] Load balancing

3. **Testing**
- [ ] Unit tests for processors
- [ ] Integration tests for multi-tenant queries
- [ ] Performance benchmarks

### Usage Examples

#### Processing Documents
```python
# Process all tenant documents
python src/main.py

# Process specific tenant
python src/main.py --tenant-id bain
```

### Querying Documents
Use the Streamlit interface to:
1. Select tenant
2. Enter query
3. View results with source citations
4. Access analytics dashboard

#### Performance Monitoring
The system includes:
- Query latency tracking
- Confidence score monitoring
- Usage analytics by tenant
- Response validation metrics

#### Security Considerations
1. Tenant isolation through Weaviate multi-tenancy
2. Role-based access control
3. API key management
4. Request validation
5. Data access logging

### Context

At Sherpa, we develop AI-enabled software applications for management consulting and professional services firms, frequently handling complex, unstructured data sources such as:

- Excel spreadsheets (Survey data, business plans, etc.)
- PDF reports
- PowerPoint slide decks
- Word documents

Our solutions must be multi-tenant, highly secure, and optimised for enterprise scalability and reliability.

## 📌 Objective

You've been provided with a set of reports, articles and presentations on the topic of artificial intelligence, sourced from top-tier consulting firms (McKinsey, Bain and BCG). The documents include a mix of data — text, tables, charts, and diagrams. Your task is to build a prototype application, within 48 hours, that allows users to explore, synthesise and interrogate the content of these reports using AI-powered techniques.

- Python is preferred, but we welcome other languages if they're your strength. Feel free to use any packages you wish
- Likewise, we can provide you with AzureOpenAI credentials, but you are also welcome to use an LLM of your choice.

The initial repository and dataset can be found in the below repo, which you create a fork of.

https://github.com/Charter-AI/sherpa-technical-task

> **Note:** We do not expect a fully-featured, enterprise grade solution. We're evaluating your approach to problem-solving, handling ambiguity, and creating robust foundations. The task is intentionally open-ended to allow you to also show off your skills. You can choose which step of the process you'd like to focus on, based on your strengths and interests.

### Deliverables

### Must-Haves

- A working codebase / repository that offers the user some ability to interact with the data. This can be as simple as the command line if you want to focus more heavily on the backend logic, but it could also be a fancy, deployed, UI if you want to show off your end-to-end development skills.
    - Please invite the following user to your repository: https://github.com/OLT2000
- README file documenting your thought process and set up instructions

### Nice-to-Haves

Aside from the basic chatbot set-up, we also place positive weightings on submissions that focus on some of the below concepts, as these are challenges that you will face on the job.

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

### Scalability ⚠️
1. **Distributed Processing with Ray**
   - Implemented Ray for parallel document processing across multiple cores
   - Ray's actor model enables efficient distribution of document chunks
   - Significantly reduces processing time for large document batches
   - Handles memory management for large PDF processing tasks

2. **Infrastructure**
   - ✅ Implemented batch processing and tenant isolation
   - ⏳ Connection pooling and caching planned
   - ⏳ Load balancing for distributed query handling

Ray was chosen for its:
- Native Python integration
- Efficient memory management for large documents
- Built-in fault tolerance
- Scalable from single machine to cluster deployment
- Simple API that works with existing Python code