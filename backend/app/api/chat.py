from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.db.mongo import get_db
from backend.app.db.vector import search_tenant_docs
from backend.app.models.schemas import ChatRequest, ChatResponse, SourceItem, ConversationMessagesResponse, MessageItem
from backend.app.services.auth import get_current_user
from backend.app.services.calculator import safe_calculate
from backend.app.services.embedding import get_embedding
from backend.app.services.llm import rewrite_query, run_rag_agent

router = APIRouter(tags=["Chat"])

async def search_kb_helper(tenant_id: str, query: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    query_vector = get_embedding(query)
    return await search_tenant_docs(tenant_id=tenant_id, query_vector=query_vector, limit=4, score_threshold=score_threshold)

async def get_verified_conversation(db, conversation_id: str, tenant_id: str, user_id: str) -> dict:
    try:
        conv_obj_id = ObjectId(conversation_id)
    except InvalidId:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    conv = await db.conversations.find_one({"_id": conv_obj_id})
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if conv.get("tenant_id") != tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Conversation belongs to a different tenant"
        )

    # Enforce conversation ownership alongside tenant_id unless explicitly authorized sharing exists
    is_owner = str(conv.get("user_id")) == str(user_id)
    is_shared = user_id in conv.get("shared_with", [])
    if not (is_owner or is_shared):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to access this conversation"
        )

    return conv

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    db = get_db()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Conversation management
    if req.conversation_id:
        conv = await get_verified_conversation(
            db=db,
            conversation_id=req.conversation_id,
            tenant_id=tenant_id,
            user_id=user_id
        )
        conversation_id = str(conv["_id"])
    else:
        conv_doc = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "created_at": now_iso
        }
        res = await db.conversations.insert_one(conv_doc)
        conversation_id = str(res.inserted_id)

    # Fetch last 6 messages for context
    cursor = db.messages.find(
        {"tenant_id": tenant_id, "conversation_id": conversation_id}
    ).sort("created_at", 1)
    
    history_records = await cursor.to_list(length=20)
    history_messages = [
        {"role": m["role"], "content": m["content"]} for m in history_records[-6:]
    ]

    # Step 1: Query rewriting
    rewrite_res = await rewrite_query(history_messages, req.message)
    if isinstance(rewrite_res, tuple):
        rewritten_q, rewrite_degraded = rewrite_res
    else:
        rewritten_q, rewrite_degraded = rewrite_res, False

    # Step 2 & 3: Agent router & Tool execution & Synthesis
    agent_res = await run_rag_agent(
        tenant_id=tenant_id,
        rewritten_query=rewritten_q,
        search_kb_func=search_kb_helper,
        calculate_func=safe_calculate
    )
    if len(agent_res) == 3:
        answer, sources, agent_degraded = agent_res
    else:
        answer, sources = agent_res
        agent_degraded = False

    is_degraded = bool(rewrite_degraded or agent_degraded)

    # Save user message
    user_msg_doc = {
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "role": "user",
        "content": req.message,
        "sources": [],
        "is_degraded": False,
        "created_at": now_iso
    }
    await db.messages.insert_one(user_msg_doc)

    source_items = [SourceItem(**s) for s in sources]

    # Save assistant message
    asst_msg_doc = {
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "role": "assistant",
        "content": answer,
        "sources": [s.model_dump() for s in source_items],
        "is_degraded": is_degraded,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.messages.insert_one(asst_msg_doc)

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=source_items,
        is_degraded=is_degraded
    )

@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def get_conversation_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    db = get_db()

    await get_verified_conversation(
        db=db,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id
    )

    cursor = db.messages.find(
        {"tenant_id": tenant_id, "conversation_id": conversation_id}
    ).sort("created_at", 1)

    messages = []
    async for msg in cursor:
        sources = [SourceItem(**s) for s in msg.get("sources", [])]
        messages.append(MessageItem(
            id=str(msg["_id"]),
            conversation_id=msg["conversation_id"],
            tenant_id=msg["tenant_id"],
            role=msg["role"],
            content=msg["content"],
            sources=sources,
            created_at=msg["created_at"],
            is_degraded=bool(msg.get("is_degraded", False))
        ))

    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        messages=messages
    )

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    db = get_db()

    conv = await get_verified_conversation(
        db=db,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
        user_id=user_id
    )

    conv_obj_id = conv["_id"]
    await db.conversations.delete_one({"_id": conv_obj_id})
    await db.messages.delete_many({"tenant_id": tenant_id, "conversation_id": conversation_id})
