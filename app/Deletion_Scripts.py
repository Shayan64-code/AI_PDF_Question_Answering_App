######for removing##### (entire database)
# Remove-Item -Recurse -Force .\data\chroma_db   (CLI)

#####for removing specific collection:   (Script)######
# from langchain.vectorstores import Chroma

# vectordb = Chroma(
#     persist_directory="chroma_db",
#     collection_name="my_collection"
# )

# vectordb.delete_collection()

#####another better approach:#####
# import chromadb

# client = chromadb.PersistentClient(path="chroma_db")
# client.list_collections()

# client.delete_collection("my_collection")
