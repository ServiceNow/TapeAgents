import os
from unittest import mock

import pytest

from tapeagents.llms.trainable import TrainableLLM, TAPEAGENTS_LLM_TOKEN


@pytest.fixture
def clean_env():
    """Clear environment variables related to API tokens for tests"""
    with mock.patch.dict(os.environ, {}, clear=True):
        yield


def test_trainable_llm_token_from_argument():
    """
    Test that TrainableLLM uses the api_token provided as an argument
    """
    # Given an explicit API token
    explicit_token = "explicit-token-value"
    
    # When initializing TrainableLLM with this token
    llm = TrainableLLM(model_name="blah", api_token=explicit_token)
    
    # Then the LLM should use the provided token
    assert llm.api_token == explicit_token


def test_trainable_llm_token_from_tapeagents_env(clean_env):
    """
    Test that TrainableLLM sources token from TAPEAGENTS_LLM_TOKEN 
    environment variable when no argument is provided
    """
    # Given the TAPEAGENTS_LLM_TOKEN environment variable is set
    env_token = "env-tapeagents-token-value"
    with mock.patch.dict(os.environ, {TAPEAGENTS_LLM_TOKEN: env_token}):
        # When initializing TrainableLLM without an explicit token
        llm = TrainableLLM(model_name="blah")
        
        # Then the LLM should use the environment variable value
        assert llm.api_token == env_token


def test_trainable_llm_token_from_openai_env(clean_env):
    """
    Test that TrainableLLM falls back to OPENAI_API_KEY environment 
    variable when no argument or TAPEAGENTS_LLM_TOKEN is provided
    """
    # Given only the OPENAI_API_KEY environment variable is set
    env_token = "env-openai-token-value"
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": env_token}):
        # When initializing TrainableLLM without an explicit token
        llm = TrainableLLM(model_name="blah")
        
        # Then the LLM should use the environment variable value
        assert llm.api_token == env_token

