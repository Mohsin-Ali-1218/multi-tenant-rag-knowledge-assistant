import asyncio
import logging
from datetime import datetime, timezone
from backend.app.db.mongo import get_db, init_mongo_indexes, close_mongo_client
from backend.app.db.vector import upsert_chunks, close_qdrant_client
from backend.app.services.auth import hash_password
from backend.app.services.embedding import chunk_text, get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multitenant_rag.seed")

TENANTS_DATA = [
    {
        "name": "Acme Legal",
        "slug": "acme-legal",
        "user": {
            "email": "admin@acmelegal.com",
            "password": "password123",
            "role": "admin"
        },
        "documents": [
            {
                "title": "Master Services Agreement Policy",
                "raw_text": "Acme Legal Master Services Agreement standard payment terms require full payment within 30 days of invoice date. Late payments incur a 1.5% monthly interest penalty. Disputes regarding billing must be submitted in writing within 15 calendar days."
            },
            {
                "title": "NDA Standard Guidelines",
                "raw_text": "Non-Disclosure Agreements for Acme Legal client engagements remain in effect for 3 years from execution date. Confidential information includes proprietary legal strategy, client financial audit records, and internal software tools."
            },
            {
                "title": "Refund and Retainer Terms",
                "raw_text": "Retainer deposits for legal counsel are non-refundable once billable work has commenced. Unused retainer funds prior to work start will be refunded at 85% of total value minus a flat $100 administrative processing fee."
            }
        ]
    },
    {
        "name": "Acme Healthcare",
        "slug": "acme-healthcare",
        "user": {
            "email": "admin@acmehealthcare.com",
            "password": "password123",
            "role": "admin"
        },
        "documents": [
            {
                "title": "HIPAA Compliance Overview",
                "raw_text": "Acme Healthcare requires all protected health information (PHI) to be encrypted at rest and in transit using AES-256 encryption. Access to electronic health records is logged and audited daily by the compliance security officer."
            },
            {
                "title": "Patient Appointment Cancellation Policy",
                "raw_text": "Patients must cancel medical appointments at least 24 hours prior to the scheduled appointment time. Late cancellations or no-shows are subject to a mandatory $50 missed appointment fee."
            },
            {
                "title": "Medical Records Request Policy",
                "raw_text": "Medical record copy requests require a signed patient release form and are processed within 5 business days. The administrative fee is $15 for the first 20 pages and $0.50 per additional page thereafter."
            }
        ]
    }
]

async def seed_data():
    logger.info("Initializing Mongo database indexes...")
    await init_mongo_indexes()
    db = get_db()

    # Clear users & tenants for clean deterministic seed
    await db.users.delete_many({})
    await db.tenants.delete_many({})

    for t_data in TENANTS_DATA:
        slug = t_data["slug"]
        existing_tenant = await db.tenants.find_one({"slug": slug})
        if existing_tenant:
            tenant_id = str(existing_tenant["_id"])
            logger.info(f"Tenant '{t_data['name']}' already exists with ID: {tenant_id}")
        else:
            tenant_doc = {
                "name": t_data["name"],
                "slug": slug,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            res = await db.tenants.insert_one(tenant_doc)
            tenant_id = str(res.inserted_id)
            logger.info(f"Created Tenant '{t_data['name']}' with ID: {tenant_id}")

        # Seed User
        u_info = t_data["user"]
        existing_user = await db.users.find_one({"tenant_id": tenant_id, "email": u_info["email"]})
        if not existing_user:
            user_doc = {
                "tenant_id": tenant_id,
                "email": u_info["email"],
                "hashed_password": hash_password(u_info["password"]),
                "role": u_info["role"],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await db.users.insert_one(user_doc)
            logger.info(f"Created User '{u_info['email']}' for Tenant ID: {tenant_id}")

        # Seed Documents & Embeddings
        for d_info in t_data["documents"]:
            existing_doc = await db.documents.find_one({"tenant_id": tenant_id, "title": d_info["title"]})
            if not existing_doc:
                chunks = chunk_text(d_info["raw_text"])
                embeddings = get_embeddings(chunks)
                now_iso = datetime.now(timezone.utc).isoformat()
                
                doc_mongo = {
                    "tenant_id": tenant_id,
                    "title": d_info["title"],
                    "raw_text": d_info["raw_text"],
                    "chunk_count": len(chunks),
                    "uploaded_at": now_iso
                }
                res = await db.documents.insert_one(doc_mongo)
                doc_id = str(res.inserted_id)

                await upsert_chunks(
                    tenant_id=tenant_id,
                    doc_id=doc_id,
                    doc_title=d_info["title"],
                    chunks=chunks,
                    embeddings=embeddings
                )
                logger.info(f"Seeded document '{d_info['title']}' for Tenant '{t_data['name']}'")

    await close_mongo_client()
    await close_qdrant_client()
    logger.info("Seeding completed successfully!")

if __name__ == "__main__":
    asyncio.run(seed_data())
