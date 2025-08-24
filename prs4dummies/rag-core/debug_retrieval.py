import logging
from rag_core import RAGCore # Assumes RAGCore is in rag_core.py

# --- Configuration ---
# !!! SET THESE TO THE PR AND QUESTION THAT IS FAILING !!!
PR_TO_DEBUG = 3
YOUR_QUESTION = f"Who commented on PR number {PR_TO_DEBUG}?"
RETRIEVER_K_VALUE = 10 # How many results to fetch
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_semantic_test():
    """Tests only the FAISS semantic retriever."""
    try:
        # --- 1. Load RAG Core ---
        print("\n" + "="*50)
        print("STEP 1: Loading FAISS index and RAG Core...")
        
        rag_core = RAGCore(vector_store_path="../indexing/vector_store")
        print("✅ RAG Core Initialized.")

        # --- 2. Test FAISS Semantic Retriever ---
        print("\n" + "="*50)
        print(f"STEP 2: Testing FAISS Semantic Retriever (k={RETRIEVER_K_VALUE})...")
        
        faiss_retriever = rag_core.vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_K_VALUE}
        )
        faiss_results = faiss_retriever.invoke(YOUR_QUESTION)
        
        print(f"\n✅ FAISS retriever returned {len(faiss_results)} documents.")
        print("--- Top Retrieved Documents ---")

        found_target_pr = False
        for i, doc in enumerate(faiss_results):
            retrieved_pr = doc.metadata.get("pr_number", "Unknown")
            is_target = "🎯" if retrieved_pr == PR_TO_DEBUG else "  "
            if is_target.strip():
                found_target_pr = True
            
            print(f"{i+1:02d}: {is_target} Retrieved PR #{retrieved_pr} "
                  f"({doc.metadata.get('document_type', 'N/A')})")

        print("="*50)

        if found_target_pr:
            print(f"\n✅ SUCCESS: The target PR #{PR_TO_DEBUG} was found in the results.")
        else:
            print(f"\n❌ FAILURE: The target PR #{PR_TO_DEBUG} was NOT found in the top {RETRIEVER_K_VALUE} results.")
            print("   This confirms the issue is with semantic relevance. The query is not 'close' enough to the target PR's content in the vector space.")

    except Exception as e:
        print(f"\nAn error occurred during the diagnostic: {e}")

if __name__ == "__main__":
    run_semantic_test()