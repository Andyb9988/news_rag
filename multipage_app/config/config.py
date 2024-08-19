import os
from dataclasses import dataclass


@dataclass
class PipelineConfiguration:
    """Dataclass that contains all required configurables for the RAG."""

    pc_index_name_news: str
    openai_embedding_model: str
    openai_llm_model: str


development: PipelineConfiguration = PipelineConfiguration(
    pc_index_name_news="all-news",
    openai_embedding_model="text-embedding-3-small",
    openai_llm_model="gpt-4o-mini",
)


production: PipelineConfiguration = PipelineConfiguration(
    pc_index_name_news="all-news",
    openai_embedding_model="text-embedding-3-small",
    openai_llm_model="gpt-4o-mini",
)


def get_pipeline_config() -> PipelineConfiguration:
    """This function returns the DAG configurables for the runtime environment based
    on the APP_ENV environment variable.

    Args:
        None
    Returns:
        PipelineConfiguration: The correct configurables for the
        runtime environment based on the APP_ENV environment
        variable.
    Raises:
        ValueError: A ValueError exception will be raised if the
        incorrect or no APP_ENV value is found
    """

    app_env: str = os.getenv("APP_ENV", "development")

    if app_env == "development":
        return development
    elif app_env == "production":
        return production
    else:
        raise ValueError(f"Pipeline configuration not found for APP_ENV: {app_env}")
