# DocsReview MCP Migration Plan

## Overview

This document outlines the migration of the DocsReview RAG system from a Streamlit-based application to a Model Context Protocol (MCP) server that integrates with Nanobot for production-ready chat orchestration and multi-channel deployment.

## Benefits of MCP Migration

### 1. **Separation of Concerns**
- **Current**: Streamlit UI + RAG logic + LLM integration all mixed in `app.py`
- **After**: Clean separation with MCP server handling RAG logic, Nanobot handling UI/chat orchestration

### 2. **Production-Ready Infrastructure**
- **Nanobot**: Handles chat orchestration, multi-model routing, front-end rendering
- **MCP Server**: Focuses solely on RAG operations and document processing
- **Scalability**: Nanobot's proven infrastructure for production deployments

### 3. **Multi-Channel Deployment**
- **Current**: Web-only via Streamlit
- **After**: Web, Slack, SMS, and other channels via Nanobot
- **Provider Agnostic**: Easy switching between LLM providers

### 4. **Rapid Prototyping**
- **YAML Configs**: Easy agent configuration and deployment
- **Template Reuse**: Leverage Nanobot's agent templates
- **Modular Design**: Mix and match RAG capabilities

## Architecture Comparison

### Current Architecture
```
Streamlit UI (app.py)
├── Document Upload/URL Input
├── RAG System Initialization
├── LangGraph Workflow
├── Chat Interface
├── Database Operations
└── External Integrations
```

### New MCP Architecture
```
Nanobot (Multi-Channel UI)
├── Chat Orchestration
├── Multi-Model Routing
├── Frontend Rendering
└── MCP Client
    └── DocsReview MCP Server
        ├── Document Processing Tools
        ├── RAG System Tools
        ├── Query Processing Tools
        └── Workflow Tools
```

## Migration Steps

### Phase 1: MCP Server Implementation ✅
- [x] Create `mcp_server.py` with core RAG tools
- [x] Implement document processing tools
- [x] Implement RAG system tools
- [x] Implement query processing tools
- [x] Add session management

### Phase 2: Nanobot Agent Configuration ✅
- [x] Create `document_analysis_agent.yaml`
- [x] Create `rag_chat_agent.yaml`
- [x] Create `multimodal_analysis_agent.yaml`
- [x] Configure MCP server integration

### Phase 3: Testing and Validation
- [ ] Test MCP server with Nanobot
- [ ] Validate all RAG functionality
- [ ] Test multi-channel deployment
- [ ] Performance testing

### Phase 4: Production Deployment
- [ ] Deploy MCP server
- [ ] Configure Nanobot agents
- [ ] Set up monitoring and logging
- [ ] Migrate existing data

## MCP Tools Mapping

### Document Processing Tools
| Current Function | MCP Tool | Description |
|-----------------|----------|-------------|
| `extract_text_from_pdf()` | `extract_document_text` | PDF/URL text extraction with OCR |
| `extract_text_from_url()` | `extract_document_text` | Web content extraction |
| `process_document()` | `process_document_rag` | RAG system initialization |

### RAG System Tools
| Current Function | MCP Tool | Description |
|-----------------|----------|-------------|
| `RAGSystem.process_document()` | `process_document_rag` | Basic RAG processing |
| `AdvancedRAGSystem.process_document()` | `process_document_rag` | Advanced RAG with ReAct |
| `MultimodalRAGSystem.process_document()` | `process_document_rag` | Multimodal processing |

### Query Processing Tools
| Current Function | MCP Tool | Description |
|-----------------|----------|-------------|
| `generate_response()` | `query_document` | RAG + LLM integration |
| `retrieve_relevant_chunks()` | `query_document` | Hybrid search |
| LangGraph workflow | `generate_executive_summary` | Executive summary generation |

### Session Management Tools
| Current Function | MCP Tool | Description |
|-----------------|----------|-------------|
| Session state management | `list_document_sessions` | List active sessions |
| Session cleanup | `clear_document_session` | Clear sessions |
| Document stats | `get_document_stats` | Get processing statistics |

## Configuration Files

### MCP Server Configuration
```python
# mcp_server.py
server = Server("docsreview-rag")
# Tools defined in list_tools()
# Handlers in call_tool()
```

### Nanobot Agent Configuration
```yaml
# nanobot_agents/document_analysis_agent.yaml
name: "Document Analysis Agent"
mcp_servers:
  - name: "docsreview-rag"
    command: "python"
    args: ["mcp_server.py"]
```

## Deployment Strategy

### 1. **Development Environment**
```bash
# Start MCP server
python mcp_server.py

# Start Nanobot with agent config
nanobot --config nanobot_agents/document_analysis_agent.yaml
```

### 2. **Production Environment**
```bash
# Docker deployment
docker-compose up -d

# Or Kubernetes deployment
kubectl apply -f k8s/
```

### 3. **Multi-Channel Deployment**
- **Web**: Nanobot web interface
- **Slack**: Slack bot integration
- **SMS**: Twilio integration
- **API**: REST API endpoints

## Data Migration

### Database Migration
- **Current**: SQLite with `regulations` table
- **Strategy**: Keep existing database, MCP server accesses directly
- **Future**: Consider migration to PostgreSQL for production

### Session State Migration
- **Current**: Streamlit session state
- **New**: MCP server session management
- **Strategy**: In-memory sessions with optional persistence

## Testing Strategy

### Unit Tests
- [ ] Test each MCP tool individually
- [ ] Test RAG system integration
- [ ] Test error handling

### Integration Tests
- [ ] Test MCP server with Nanobot
- [ ] Test multi-agent workflows
- [ ] Test document processing pipeline

### End-to-End Tests
- [ ] Test complete document analysis workflow
- [ ] Test chat functionality
- [ ] Test multi-channel deployment

## Monitoring and Logging

### MCP Server Monitoring
- Tool execution metrics
- RAG system performance
- Error rates and types
- Session management stats

### Nanobot Monitoring
- Agent performance
- Multi-model routing efficiency
- Channel-specific metrics
- User interaction patterns

## Rollback Plan

### If Issues Arise
1. **Immediate**: Switch back to Streamlit app
2. **Short-term**: Fix MCP server issues
3. **Long-term**: Gradual migration with feature parity

### Data Consistency
- Database remains unchanged
- Session data can be reconstructed
- No data loss during migration

## Future Enhancements

### Advanced Features
- [ ] Real-time document collaboration
- [ ] Advanced analytics and insights
- [ ] Custom agent workflows
- [ ] Integration with more MCP servers

### Scalability Improvements
- [ ] Horizontal scaling of MCP servers
- [ ] Load balancing
- [ ] Caching strategies
- [ ] Performance optimization

## Conclusion

The MCP migration provides a clean, scalable architecture that separates concerns while leveraging Nanobot's production-ready infrastructure. This enables multi-channel deployment, better maintainability, and rapid prototyping of new RAG capabilities.

The migration maintains all existing functionality while providing a foundation for future enhancements and integrations.