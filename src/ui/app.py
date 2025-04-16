import streamlit as st
import requests
import json

# Constants
API_URL = "http://127.0.0.1:8000"
TENANTS = ["bain", "bcg", "mck"]

st.set_page_config(page_title="Consulting Reports Q&A", page_icon="📚", layout="wide")

st.title("Consulting Reports Q&A")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    tenant_id = st.selectbox(
        "Select Tenant",
        options=TENANTS,
        index=0,
        help="Choose which tenant's documents to query",
    )

    max_results = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of relevant document chunks to retrieve",
    )

# Main query interface
query = st.text_area(
    "Enter your question",
    placeholder="e.g., What are the key trends in luxury retail?",
    height=100,
)

if st.button("Submit Query", type="primary"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing query..."):
            try:
                # Call the API
                response = requests.post(
                    f"{API_URL}/query",
                    headers={"tenant-id": tenant_id},
                    json={"query": query, "max_results": max_results},
                )
                response.raise_for_status()
                result = response.json()

                # Display answer
                st.markdown("### Answer")
                st.write(result["answer"])

                # Display sources
                st.markdown("### Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}: {source['document_id']}"):
                        st.json(source["metadata"])
                        st.metric(
                            "Relevance Score", f"{(1 - source['distance']) * 100:.1f}%"
                        )

                # Display context
                st.markdown("### Context")
                for i, context in enumerate(result["context"], 1):
                    with st.expander(f"Context {i}"):
                        st.markdown(context)

            except requests.exceptions.RequestException as e:
                st.error(f"Error calling API: {str(e)}")
            except Exception as e:
                st.error(f"Error processing result: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    💡 **Tips:**
    - Be specific in your questions
    - Try asking about specific industries or trends
    - You can compare insights across different consulting firms
    """
)
