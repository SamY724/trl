# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Environments for agentic reinforcement learning.

This module provides a flexible and extensible system for creating AI agent environments
that can interact with various tools and execution backends. The environments support
code execution, tool calling, and other agentic behaviors during text generation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from transformers import PreTrainedTokenizerBase

from ..extras.vllm_client import VLLMClientGenerationConfig


logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """
    Base configuration class for all environments.
    
    This class provides common configuration options that all environments can use.
    Specific environment implementations should subclass this to add their own
    configuration parameters.
    
    Args:
        max_turns (`int`, *optional*, defaults to `10`):
            Maximum number of generation turns allowed per conversation.
        timeout_seconds (`float`, *optional*, defaults to `30.0`):
            Maximum time in seconds to spend on each turn.
        error_on_timeout (`bool`, *optional*, defaults to `False`):
            Whether to raise an error when timeout is reached, or just return the current state.
    """
    
    max_turns: int = field(default=10, metadata={"help": "Maximum number of generation turns allowed per conversation."})
    timeout_seconds: float = field(default=30.0, metadata={"help": "Maximum time in seconds to spend on each turn."})
    error_on_timeout: bool = field(default=False, metadata={"help": "Whether to raise an error when timeout is reached."})


# Import the ExecutorProtocol from executors module
try:
    from .executors import ExecutorProtocol
except ImportError:
    # Fallback protocol definition if executors module is not available
    class ExecutorProtocol(Protocol):
        """
        Protocol defining the interface for code executors.
        
        Any executor used with environments must implement this interface.
        """
        
        def execute(self, code_blocks: List[str]) -> List[str]:
            """
            Execute a list of code blocks and return their outputs.
            
            Args:
                code_blocks: List of code strings to execute.
                
            Returns:
                List of execution results as strings.
            """
            ...


class Environment(ABC):
    """
    Base environment class for agentic reinforcement learning.
    
    This class provides the core interface that all environments must implement.
    Environments handle the interaction between language models and various tools
    or execution backends during text generation.
    
    The base class defines the essential methods for generation and provides
    common utilities for configuration management and error handling.
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """
        Initialize the environment with optional configuration.
        
        Args:
            config: Environment configuration. If None, uses default configuration.
        """
        self.config = config or EnvironmentConfig()
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def generate(
        self, 
        vllm_client: Any, 
        generation_config: VLLMClientGenerationConfig, 
        prompts: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate responses using VLLM client.
        
        This is the main method that environments must implement. It should handle
        the complete generation process, including any tool interactions or special
        processing required by the specific environment.
        
        Args:
            vllm_client: VLLM client instance for text generation.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.
            
        Returns:
            Tuple containing:
                - completion_ids: Generated token IDs (list of lists)
                - completion_mask: Mask indicating which tokens to consider for training (list of lists)
        """
        raise NotImplementedError
        
    def validate_inputs(
        self, 
        vllm_client: Any, 
        generation_config: VLLMClientGenerationConfig, 
        prompts: List[str]
    ) -> None:
        """
        Validate inputs before generation.
        
        Args:
            vllm_client: VLLM client instance.
            generation_config: Generation configuration.
            prompts: Input prompts.
            
        Raises:
            ValueError: If inputs are invalid.
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        if generation_config is None:
            raise ValueError("Generation config must be provided")
        if not hasattr(vllm_client, 'generate'):
            raise ValueError("VLLM client must have a 'generate' method")
            
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Environment":
        """
        Create an environment instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Environment instance.
        """
        env_config = EnvironmentConfig(**config.get('environment_config', {}))
        return cls(config=env_config)


@dataclass
class CodeAgentConfig(EnvironmentConfig):
    """
    Configuration for CodeAgentEnvironment.
    
    Extends the base EnvironmentConfig with code execution specific parameters.
    
    Args:
        parsing_string (`str`, *optional*, defaults to `"<code>"`):
            String that marks the beginning of code blocks in generated text.
        stop_string (`str`, *optional*, defaults to `"</code>"`):
            String that marks the end of code blocks in generated text.
        tools_script (`Optional[str]`, *optional*, defaults to `None`):
            Optional script to prepend to each extracted code block before execution.
        output_string_start (`str`, *optional*, defaults to `"<output>"`):
            String marking the beginning of code output in conversation.
        output_string_end (`str`, *optional*, defaults to `"</output>"`):
            String marking the end of code output in conversation.
        max_tokens_per_turn (`int`, *optional*, defaults to `1024`):
            Maximum tokens to generate per turn to prevent runaway generation.
    """
    
    parsing_string: str = field(default="<code>", metadata={"help": "String that marks the beginning of code blocks."})
    stop_string: str = field(default="</code>", metadata={"help": "String that marks the end of code blocks."})
    tools_script: Optional[str] = field(default=None, metadata={"help": "Optional script to prepend to extracted code."})
    output_string_start: str = field(default="<output>", metadata={"help": "String marking the beginning of code output."})
    output_string_end: str = field(default="</output>", metadata={"help": "String marking the end of code output."})
    max_tokens_per_turn: int = field(default=1024, metadata={"help": "Maximum tokens to generate per turn."})


class DefaultEnvironment(Environment):
    """
    Default environment that implements standard VLLM generation without any special processing.
    
    This environment performs straightforward text generation using the VLLM client
    and creates a mask that includes all generated tokens for training.
    """
    
    def generate(
        self, 
        vllm_client: Any, 
        generation_config: VLLMClientGenerationConfig, 
        prompts: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate responses using VLLM without any special processing.
        
        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.
            
        Returns:
            Tuple containing:
                - completion_ids: Generated token IDs
                - completion_mask: Mask with 1s for all tokens
        """
        self.validate_inputs(vllm_client, generation_config, prompts)
        
        try:
            completion_ids = vllm_client.generate(
                prompts=prompts,
                **vars(generation_config),
            )
            
            # Create a mask with all 1s matching the shape of completion_ids
            completion_mask = [[1] * len(ids) for ids in completion_ids]
            
            logger.debug(f"Generated {len(completion_ids)} completions with lengths: {[len(ids) for ids in completion_ids]}")
            
            return completion_ids, completion_mask
            
        except Exception as e:
            logger.error(f"Generation failed in DefaultEnvironment: {e}")
            raise


class CodeParser:
    """
    Handles parsing and extraction of code blocks from generated text.
    """
    
    def __init__(self, config: CodeAgentConfig):
        self.config = config
    
    def extract_code(self, text: str) -> Optional[str]:
        """
        Extract code from the last code block in the generated text.
        
        Args:
            text: Generated text potentially containing code blocks.
            
        Returns:
            Extracted code string or None if no code found.
        """
        if self.config.parsing_string not in text:
            return None

        # Find the last occurrence of the parsing string
        last_code_start_index = text.rfind(self.config.parsing_string)
        if last_code_start_index == -1:
            return None

        code_segment = text[last_code_start_index + len(self.config.parsing_string):]

        # Find the first occurrence of the stop string after the last parsing string
        stop_index = code_segment.find(self.config.stop_string)
        if stop_index != -1:
            code_parts = code_segment[:stop_index]
        else:
            # If stop string is not found, use everything after parsing string
            code_parts = code_segment

        # Prepend tools script if available
        if self.config.tools_script:
            code_parts = f"{self.config.tools_script}\n{code_parts}"

        return code_parts if code_parts.strip() else None


class TokenMasker:
    """
    Handles creation of token masks for training, masking tool outputs.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: CodeAgentConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def create_masks(
        self,
        completion_ids_list: List[List[int]],
        output_string_start: Optional[str] = None,
        output_string_end: Optional[str] = None,
    ) -> List[List[int]]:
        """
        Create masks for token sequences, marking tokens that are part of tool outputs with 0s.
        
        Args:
            completion_ids_list: List of token ID sequences.
            output_string_start: Override for output start string.
            output_string_end: Override for output end string.
            
        Returns:
            List of mask lists (1s for tokens to keep, 0s for tokens to mask).
        """
        current_output_start = output_string_start or self.config.output_string_start
        current_output_end = output_string_end or self.config.output_string_end
        
        # Validate and get token IDs for start/end markers
        start_token_id = self._get_single_token_id(current_output_start, "output_string_start")
        end_token_id = self._get_single_token_id(current_output_end, "output_string_end")
        
        completion_masks = []
        for completion_ids in completion_ids_list:
            mask = [1] * len(completion_ids)
            if not completion_ids:
                completion_masks.append(mask)
                continue
                
            self._apply_mask_for_outputs(completion_ids, mask, start_token_id, end_token_id)
            completion_masks.append(mask)
            
        return completion_masks
    
    def _get_single_token_id(self, text: str, param_name: str) -> int:
        """Get token ID for text, ensuring it maps to exactly one token."""
        if not text.strip():
            raise ValueError(f"{param_name} is empty or whitespace.")
        
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"{param_name} '{text}' does not correspond to a single token. "
                f"Encoded to: {encoded}. Ensure it's a single token."
            )
        return encoded[0]
    
    def _apply_mask_for_outputs(
        self, 
        completion_ids: List[int], 
        mask: List[int], 
        start_token_id: int, 
        end_token_id: int
    ) -> None:
        """Apply masking for output sections in the completion."""
        i = 0
        while i < len(completion_ids):
            if completion_ids[i] == start_token_id:
                # Found start marker, look for end marker
                start_idx = i
                j = i + 1
                found_end = False
                
                while j < len(completion_ids):
                    if completion_ids[j] == end_token_id:
                        # Mask from start to end (inclusive)
                        for k in range(start_idx, j + 1):
                            if k < len(mask):
                                mask[k] = 0
                        i = j + 1
                        found_end = True
                        break
                    j += 1
                
                if not found_end:
                    i += 1
            else:
                i += 1


class ConversationTracker:
    """
    Tracks conversation state during multi-turn generation with code execution.
    """
    
    def __init__(self, prompts: List[str], generation_config: VLLMClientGenerationConfig, tokenizer: PreTrainedTokenizerBase):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.total_max_tokens = generation_config.max_tokens
        
        # Track conversations by (prompt_idx, copy_idx)
        self.conversation_map: Dict[Tuple[int, int], str] = {}
        self.active_conversations: List[str] = []
        self.active_conversations_metadata: List[Tuple[int, int]] = []
        self.prompt_lengths: Dict[Tuple[int, int], int] = {}
        
        # Initialize active conversations
        self._initialize_conversations(generation_config.n)
    
    def _initialize_conversations(self, n_copies: int) -> None:
        """Initialize conversations from prompts."""
        for prompt_idx, prompt in enumerate(self.prompts):
            for copy_idx in range(n_copies):
                metadata = (prompt_idx, copy_idx)
                self.active_conversations.append(prompt)
                self.active_conversations_metadata.append(metadata)
                self.prompt_lengths[metadata] = len(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
    
    def has_active_conversations(self) -> bool:
        """Check if there are any active conversations."""
        return len(self.active_conversations) > 0
    
    def complete_conversation(self, metadata: Tuple[int, int], conversation: str) -> None:
        """Mark a conversation as completed."""
        self.conversation_map[metadata] = conversation
        self._remove_active_conversation(metadata)
    
    def continue_conversation(self, metadata: Tuple[int, int], conversation: str) -> None:
        """Continue an active conversation."""
        # Find and update the conversation
        for i, active_metadata in enumerate(self.active_conversations_metadata):
            if active_metadata == metadata:
                self.active_conversations[i] = conversation
                break
    
    def exceeds_token_limit(self, metadata: Tuple[int, int], conversation: str) -> bool:
        """Check if conversation exceeds token limits."""
        total_tokens = len(self.tokenizer.encode(conversation, add_special_tokens=False))
        completion_tokens = total_tokens - self.prompt_lengths[metadata]
        return completion_tokens > self.total_max_tokens
    
    def truncate_conversation(self, metadata: Tuple[int, int], conversation: str) -> str:
        """Truncate conversation to stay within token limits."""
        # Find original prompt
        original_prompt = ""
        for prompt in self.prompts:
            if conversation.startswith(prompt):
                original_prompt = prompt
                break
        
        # Encode and truncate
        prompt_tokens = self.tokenizer.encode(original_prompt, add_special_tokens=False)
        full_tokens = self.tokenizer.encode(conversation, add_special_tokens=False)
        truncate_at = len(prompt_tokens) + self.total_max_tokens
        truncated_tokens = full_tokens[:truncate_at]
        
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
    
    def _remove_active_conversation(self, metadata: Tuple[int, int]) -> None:
        """Remove conversation from active list."""
        for i, active_metadata in enumerate(self.active_conversations_metadata):
            if active_metadata == metadata:
                del self.active_conversations[i]
                del self.active_conversations_metadata[i]
                break
    
    def get_completed_conversations(self) -> List[str]:
        """Get completed conversations in original order."""
        completed = []
        for prompt_idx in range(len(self.prompts)):
            for copy_idx in range(len([m for m in self.conversation_map.keys() if m[0] == prompt_idx])):
                metadata = (prompt_idx, copy_idx)
                if metadata in self.conversation_map:
                    completed.append(self.conversation_map[metadata])
        return completed


class CodeAgentEnvironment(Environment):
    """
    Environment that supports code execution during generation.

    This environment enables an agent to generate text that includes code blocks,
    execute those code blocks using a provided code executor, and insert the 
    execution results back into the conversation.

    The environment uses a multi-turn generation approach where it:
    1. Generates text until it encounters a code block
    2. Extracts and executes the code
    3. Appends the execution result to the conversation
    4. Continues generation until no more code blocks are found

    Args:
        code_executor: An object implementing ExecutorProtocol for running code.
        tokenizer: Tokenizer for encoding and decoding text.
        config: Configuration for the code agent environment.
    """

    def __init__(
        self,
        code_executor: ExecutorProtocol,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[CodeAgentConfig] = None,
    ):
        """
        Initialize the code agent environment.

        Args:
            code_executor: The executor to run code blocks.
            tokenizer: Tokenizer for encoding/decoding text.
            config: Configuration for the environment.
        """
        super().__init__(config or CodeAgentConfig())
        
        if not hasattr(code_executor, "execute"):
            raise ValueError("code_executor must have an 'execute' method.")

        self.code_executor = code_executor
        self.tokenizer = tokenizer
        self.parser = CodeParser(self.config)
        self.masker = TokenMasker(tokenizer, self.config)
        
        logger.info(f"Initialized CodeAgentEnvironment with parsing: '{self.config.parsing_string}' -> '{self.config.stop_string}'")

    def run_agent(
        self, 
        vllm_client: Any, 
        generation_config: VLLMClientGenerationConfig, 
        prompts: List[str]
    ) -> List[str]:
        """
        Run the agent with code execution and return completed text responses.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            List of completed text responses with code execution results.
        """
        self.validate_inputs(vllm_client, generation_config, prompts)
        
        conversation_tracker = ConversationTracker(prompts, generation_config, self.tokenizer)
        step_config = self._create_step_generation_config(generation_config)
        
        for turn in range(self.config.max_turns):
            if not conversation_tracker.has_active_conversations():
                break
                
            logger.debug(f"Turn {turn + 1}: Processing {len(conversation_tracker.active_conversations)} conversations")
            
            # Generate next step for all active conversations
            outputs = vllm_client.generate(
                prompts=conversation_tracker.active_conversations,
                **vars(step_config)
            )
            
            # Process outputs and handle code execution
            code_batch, pending_conversations = self._process_generation_outputs(
                outputs, conversation_tracker
            )
            
            # Execute code if any was found
            if code_batch:
                self._execute_code_batch(code_batch, pending_conversations, conversation_tracker)
        
        return conversation_tracker.get_completed_conversations()
        
    def _create_step_generation_config(self, generation_config: VLLMClientGenerationConfig) -> VLLMClientGenerationConfig:
        """Create generation config for individual steps."""
        stop_sequences = [self.config.stop_string]
        if generation_config.stop:
            stop_sequences = list(set(stop_sequences + generation_config.stop))

        return VLLMClientGenerationConfig(
            n=1,
            repetition_penalty=generation_config.repetition_penalty,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            min_p=generation_config.min_p,
            max_tokens=self.config.max_tokens_per_turn,
            guided_decoding_regex=generation_config.guided_decoding_regex,
            stop=stop_sequences,
        )
        
    def _process_generation_outputs(
        self, 
        outputs: List[Any], 
        tracker: 'ConversationTracker'
    ) -> Tuple[List[str], List[Tuple[str, Any]]]:
        """Process generation outputs and identify code blocks for execution."""
        code_batch = []
        pending_conversations = []
        
        for i, generated_token_ids in enumerate(outputs):
            current_conversation = tracker.active_conversations[i]
            metadata = tracker.active_conversations_metadata[i]
            
            if not isinstance(generated_token_ids, list):
                tracker.complete_conversation(metadata, current_conversation)
                continue
                
            # Decode and process the generation
            generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            full_conversation = current_conversation + generated_text
            
            # Check for token limits and truncate if necessary
            if tracker.exceeds_token_limit(metadata, full_conversation):
                truncated = tracker.truncate_conversation(metadata, full_conversation)
                tracker.complete_conversation(metadata, truncated)
                continue
            
            # Check for code execution
            if self._should_execute_code(generated_text, full_conversation, current_conversation):
                code = self.parser.extract_code(full_conversation)
                if code:
                    code_batch.append(code)
                    pending_conversations.append((full_conversation, metadata))
                else:
                    tracker.complete_conversation(metadata, full_conversation)
            else:
                tracker.complete_conversation(metadata, full_conversation)
                
        return code_batch, pending_conversations
    
    def _should_execute_code(self, generated_text: str, full_conversation: str, current_conversation: str) -> bool:
        """Check if code execution should be performed."""
        stopped_by_code_tag = generated_text.rstrip().endswith(self.config.stop_string.rstrip())
        last_code_start = full_conversation.rfind(self.config.parsing_string)
        is_code_in_new_text = (
            last_code_start != -1 and 
            last_code_start >= len(current_conversation)
        )
        return is_code_in_new_text and stopped_by_code_tag
    
    def _execute_code_batch(
        self, 
        code_batch: List[str], 
        pending_conversations: List[Tuple[str, Any]], 
        tracker: 'ConversationTracker'
    ) -> None:
        """Execute a batch of code and update conversations."""
        try:
            execution_results = self.code_executor.execute(code_batch)
            if len(execution_results) != len(pending_conversations):
                raise ValueError(
                    f"Mismatch between code batch size ({len(code_batch)}) and results ({len(execution_results)})"
                )
            
            # Update conversations with execution results
            for (conversation, metadata), result in zip(pending_conversations, execution_results):
                updated_conversation = (
                    f"{conversation}{self.config.output_string_start}{result}{self.config.output_string_end}"
                )
                tracker.continue_conversation(metadata, updated_conversation)
                
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            # Mark all pending conversations as completed on error
            for conversation, metadata in pending_conversations:
                tracker.complete_conversation(metadata, conversation)

    def generate(
        self, 
        vllm_client: Any, 
        generation_config: VLLMClientGenerationConfig, 
        prompts: List[str]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate responses with code execution and return token IDs of the completions.
        
        This method runs the full agent workflow with code execution and returns
        both the completion token IDs and appropriate masks for training.

        Args:
            vllm_client: VLLM client instance.
            generation_config: Configuration for generation parameters.
            prompts: Input prompts for generation.

        Returns:
            Tuple containing:
                - completion_ids: Generated token ID lists (completions only)
                - completion_mask: Masks for training (0s mask tool outputs)
        """
        # Get completed text responses from the agent logic
        completed_conversations = self.run_agent(vllm_client, generation_config, prompts)

        completion_ids = []
        for final_output in completed_conversations:
            # Extract completion text by finding the original prompt
            completion_text = self._extract_completion_text(final_output, prompts)
            
            # Convert to token IDs
            completion_token_ids = self.tokenizer.encode(completion_text, add_special_tokens=True)
            completion_ids.append(completion_token_ids)

        # Generate masks for the completions using the modular masker
        completion_mask = self.masker.create_masks(completion_ids)

        logger.debug(f"Generated {len(completion_ids)} completions with masks")
        return completion_ids, completion_mask
    
    def _extract_completion_text(self, final_output: str, prompts: List[str]) -> str:
        """Extract completion text from final output by removing the original prompt."""
        # Find matching prompt and extract everything after it
        for prompt in prompts:
            if final_output.startswith(prompt):
                return final_output[len(prompt):]
        
        # Fallback: check if prompt is anywhere in the output
        for prompt in prompts:
            if prompt in final_output:
                start_index = final_output.find(prompt) + len(prompt)
                return final_output[start_index:]
        
        logger.warning("No matching prompt found in completion. Using full output.")
        return final_output
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CodeAgentEnvironment":
        """
        Create a CodeAgentEnvironment from a configuration dictionary.
        
        Args:
            config: Configuration dictionary containing:
                - code_executor: The code executor instance
                - tokenizer: The tokenizer instance  
                - environment_config: Environment configuration parameters
                
        Returns:
            CodeAgentEnvironment instance.
        """
        code_executor = config["code_executor"]
        tokenizer = config["tokenizer"]
        env_config = CodeAgentConfig(**config.get("environment_config", {}))
        
        return cls(
            code_executor=code_executor,
            tokenizer=tokenizer,
            config=env_config
        )


# Environment Registry System
class EnvironmentRegistry:
    """
    Registry for environment types, enabling easy discovery and instantiation.
    
    This registry allows users to register custom environments and create them
    by name, following common ML framework patterns.
    """
    
    _environments: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, environment_class: type) -> None:
        """
        Register an environment class with a given name.
        
        Args:
            name: Name to register the environment under.
            environment_class: Environment class to register.
        """
        if not issubclass(environment_class, Environment):
            raise ValueError(f"Environment class {environment_class} must inherit from Environment")
        
        cls._environments[name] = environment_class
        logger.info(f"Registered environment: {name} -> {environment_class.__name__}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Environment:
        """
        Create an environment instance by name.
        
        Args:
            name: Name of the registered environment.
            **kwargs: Arguments to pass to the environment constructor.
            
        Returns:
            Environment instance.
            
        Raises:
            ValueError: If environment name is not registered.
        """
        if name not in cls._environments:
            raise ValueError(
                f"Environment '{name}' not found. Available environments: {list(cls._environments.keys())}"
            )
        
        environment_class = cls._environments[name]
        return environment_class(**kwargs)
    
    @classmethod
    def list_environments(cls) -> List[str]:
        """
        List all registered environment names.
        
        Returns:
            List of registered environment names.
        """
        return list(cls._environments.keys())
    
    @classmethod
    def get_environment_class(cls, name: str) -> type:
        """
        Get environment class by name.
        
        Args:
            name: Environment name.
            
        Returns:
            Environment class.
        """
        if name not in cls._environments:
            raise ValueError(f"Environment '{name}' not registered")
        return cls._environments[name]


# Register built-in environments
EnvironmentRegistry.register("default", DefaultEnvironment)
EnvironmentRegistry.register("code_agent", CodeAgentEnvironment)


# Convenience functions for common use cases
def create_environment(name: str, **kwargs) -> Environment:
    """
    Create an environment by name.
    
    Args:
        name: Environment name (e.g., 'default', 'code_agent').
        **kwargs: Environment-specific arguments.
        
    Returns:
        Environment instance.
    """
    return EnvironmentRegistry.create(name, **kwargs)


def list_available_environments() -> List[str]:
    """
    List all available environment types.
    
    Returns:
        List of environment names.
    """
    return EnvironmentRegistry.list_environments()


def register_environment(name: str, environment_class: type) -> None:
    """
    Register a custom environment.
    
    Args:
        name: Name for the environment.
        environment_class: Environment class to register.
    """
    EnvironmentRegistry.register(name, environment_class)