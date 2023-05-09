from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    "https://3aca97bc-6507-44f6-b86f-291a3a415e0b.us-east-1-0.aws.cloud.qdrant.io", 
    prefer_grpc=True,
    api_key="5oNuOUCUKR9hTqp4sCCP8tyUP_OnzgceVKED4sKQfp65F0VvUmG9kA",
)

print(qdrant_client.get_collections())
