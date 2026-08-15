import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai import types
from backend.app.config import settings

logger = logging.getLogger("multitenant_rag.llm")

def get_genai_client() -> genai.Client:
    return genai.Client(api_key=settings.GEMINI_API_KEY)

async def rewrite_query(chat_history: List[Dict[str, str]], latest_message: str) -> str:
    """
    Rewrites the latest user message into a standalone query using conversation history.
    Logs both original and rewritten query.
    """
    if not chat_history or not settings.GEMINI_API_KEY:
        logger.info(f"Query rewrite [Fast path] Original: '{latest_message}' | Rewritten: '{latest_message}'")
        return latest_message

    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    prompt = f"""Given the following conversation history and a follow-up user message, rewrite the follow-up message into a self-contained, standalone question or request that includes all necessary context from history.

Conversation History:
{history_str}

Follow-up User Message:
{latest_message}

Rules:
- Do NOT answer the question.
- Output ONLY the rewritten standalone question.
- Preserve exact names, numbers, or specific terms.
"""

    try:
        client = get_genai_client()
        # Non-blocking async execution with 3 second timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.5-flash",
                contents=prompt
            ),
            timeout=3.0
        )
        rewritten = response.text.strip() if response.text else latest_message
        logger.info(f"Query rewrite | Original: '{latest_message}' | Rewritten: '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewriting failed with error: {e}. Falling back to original message.")
        return latest_message

tool_search_kb = types.FunctionDeclaration(
    name="search_knowledge_base",
    description="Search the tenant's knowledge base for relevant documents, policies, or FAQs.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="The search query or keyword phrase to find matching documents."
            )
        },
        required=["query"]
    )
)

tool_calculate = types.FunctionDeclaration(
    name="calculate",
    description="Evaluate arithmetic mathematical expressions safely (e.g., '450 * 0.15' or '1000 - 250').",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "expression": types.Schema(
                type=types.Type.STRING,
                description="The arithmetic mathematical expression to evaluate."
            )
        },
        required=["expression"]
    )
)

agent_tools = types.Tool(function_declarations=[tool_search_kb, tool_calculate])

async def run_rag_agent(
    tenant_id: str,
    rewritten_query: str,
    search_kb_func: Any,
    calculate_func: Any,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Agent router function that:
    1. Asks Gemini which tool(s) to run (search_knowledge_base, calculate, or direct answer).
    2. Executes requested tools (can call both search & calc in one turn).
    3. Synthesizes final response with cited sources.
    """
    sources: List[Dict[str, Any]] = []
    tool_results: List[str] = []

    if not settings.GEMINI_API_KEY:
        logger.info("No GEMINI_API_KEY configured. Executing direct RAG retrieval & tool processing.")
        kb_hits = await search_kb_func(tenant_id, rewritten_query)
        sources.extend(kb_hits)
        
        calc_result_msg = ""
        import re
        math_match = re.search(r'([\d\.]+\s*[\+\-\*/]\s*[\d\.]+)', rewritten_query)
        if math_match:
            expr = math_match.group(1)
            calc_val = calculate_func(expr)
            calc_result_msg = f"\n\nCalculated result for ({expr}): {calc_val}"

        if kb_hits:
            doc_context = "\n\n".join([f"[{h['document_title']}]\n{h['chunk_text']}" for h in kb_hits])
            answer = f"Based on knowledge base documents:\n{doc_context}{calc_result_msg}"
        else:
            answer = f"No matching documents found in knowledge base.{calc_result_msg}"

        unique_sources = []
        seen = set()
        for src in sources:
            key = (src.get("document_title"), src.get("chunk_text"))
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "document_title": src.get("document_title"),
                    "chunk_text": src.get("chunk_text"),
                    "score": src.get("score", 0.0)
                })

        return answer, unique_sources

    try:
        client = get_genai_client()
        config = types.GenerateContentConfig(tools=[agent_tools])
        
        # Step 1: Initial agent call with non-blocking 3.0s timeout
        initial_response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model="gemini-3.5-flash",
                contents=f"User Query: {rewritten_query}",
                config=config
            ),
            timeout=3.0
        )

        function_calls = initial_response.function_calls or []
        
        if not function_calls:
            query_lower = rewritten_query.lower()
            if any(k in query_lower for k in ["calculate", "math", "sum", "refund for", "amount", "%", "+", "-", "*", "/"]) and any(c.isdigit() for c in query_lower):
                kb_hits = await search_kb_func(tenant_id, rewritten_query)
                if kb_hits:
                    sources.extend(kb_hits)
                    formatted_hits = "\n".join([f"- Doc: {h['document_title']}\n  Content: {h['chunk_text']}" for h in kb_hits])
                    tool_results.append(f"Knowledge Base Search Results:\n{formatted_hits}")
            else:
                kb_hits = await search_kb_func(tenant_id, rewritten_query)
                if kb_hits:
                    sources.extend(kb_hits)
                    formatted_hits = "\n".join([f"- Doc: {h['document_title']}\n  Content: {h['chunk_text']}" for h in kb_hits])
                    tool_results.append(f"Knowledge Base Search Results:\n{formatted_hits}")

        for call in function_calls:
            if call.name == "search_knowledge_base":
                search_q = call.args.get("query", rewritten_query)
                hits = await search_kb_func(tenant_id, search_q)
                sources.extend(hits)
                formatted_hits = "\n".join([f"- Doc: {h['document_title']}\n  Content: {h['chunk_text']}" for h in hits])
                tool_results.append(f"Tool 'search_knowledge_base' results for query '{search_q}':\n{formatted_hits}")
            elif call.name == "calculate":
                expr = call.args.get("expression", "")
                calc_res = calculate_func(expr)
                tool_results.append(f"Tool 'calculate' result for '{expr}': {calc_res}")

        # Step 2: Final synthesis
        if tool_results:
            tool_context = "\n\n".join(tool_results)
            synthesis_prompt = f"""You are a helpful Multi-Tenant RAG Knowledge Assistant. Answer the user's query accurately using ONLY the provided tool results. 

Original User Query: {rewritten_query}

Tool Execution Results:
{tool_context}

Please provide a clear, helpful response. Cite relevant document titles if knowledge base information was used.
"""
            final_resp = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-3.5-flash",
                    contents=synthesis_prompt
                ),
                timeout=3.0
            )
            answer = final_resp.text.strip() if final_resp.text else "I completed the processing but received no text response."
        else:
            answer = initial_response.text.strip() if initial_response.text else "I am unable to answer this query based on available tools."

        unique_sources = []
        seen = set()
        for src in sources:
            key = (src.get("document_title"), src.get("chunk_text"))
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "document_title": src.get("document_title"),
                    "chunk_text": src.get("chunk_text"),
                    "score": src.get("score", 0.0)
                })

        return answer, unique_sources

    except Exception as e:
        logger.error(f"Agent execution failed with error: {e}. Executing deterministic fallback.", exc_info=True)
        kb_hits = await search_kb_func(tenant_id, rewritten_query)
        sources.extend(kb_hits)
        
        calc_result_msg = ""
        import re
        math_match = re.search(r'([\d\.]+\s*[\+\-\*/]\s*[\d\.]+)', rewritten_query)
        if math_match:
            expr = math_match.group(1)
            calc_val = calculate_func(expr)
            calc_result_msg = f"\n\nCalculated result for ({expr}): {calc_val}"

        if kb_hits:
            doc_context = "\n\n".join([f"[{h['document_title']}]\n{h['chunk_text']}" for h in kb_hits])
            answer = f"Based on knowledge base documents:\n{doc_context}{calc_result_msg}"
        else:
            answer = f"No matching documents found in knowledge base.{calc_result_msg}"

        unique_sources = []
        seen = set()
        for src in sources:
            key = (src.get("document_title"), src.get("chunk_text"))
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "document_title": src.get("document_title"),
                    "chunk_text": src.get("chunk_text"),
                    "score": src.get("score", 0.0)
                })

        return answer, unique_sources
