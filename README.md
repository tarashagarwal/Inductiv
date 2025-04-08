#Inductiv

Inductiv is an experimental tool currently under development, designed to create a general-purpose recommendation system. The goal is to support training on a generic format of input datasets and leverage features stored in a local database.

How It Works

    Features are extracted from the input dataset and stored locally.

    These features are then pulled from the database, trained on the dataset, and transformed into vector representations.

    The resulting vectors are stored in a vector database for efficient retrieval.

    A conversational interface is provided using an LLM (e.g., LLaMA or other models), which can be fine-tuned on specific datasets.

    When a user asks a question, the system performs a RAG (Retrieval-Augmented Generation) operation, retrieving relevant vector data based on similarity and generating a contextual response.

Key Features

    Generic dataset format support

    Local feature and vector database integration

    LLM-based conversational interface

    Retrieval-Augmented Generation (RAG) for intelligent querying

    Easily extendable and fine-tunable