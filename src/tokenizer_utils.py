import tiktoken
from typing import List, Union, Dict


class TiktokenHuggingFaceTokenizer:
    def __init__(self, model_name: str):
        """
        Initialize the tokenizer with the specified OpenAI model.

        Args:
            model_name (str): The name of the OpenAI model (e.g., 'gpt-3.5-turbo').
        """
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.model_name = model_name

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encode a single string into token IDs.

        Args:
            text (str): The text to encode.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List[int]: A list of token IDs.
        """
        return self.encoding.encode(text)

    def decode(
        self, tokens: Union[List[int], List[List[int]]], **kwargs
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back into a string.

        Args:
            tokens (List[int] or List[List[int]]): Token IDs to decode.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            str or List[str]: The decoded string(s).
        """
        if isinstance(tokens[0], list):
            return [self.encoding.decode(token_list) for token_list in tokens]
        return self.encoding.decode(tokens)

    def batch_encode(self, texts: List[str], **kwargs) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.

        Args:
            texts (List[str]): The list of texts to encode.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List[List[int]]: A list of lists containing token IDs.
        """
        return [self.encoding.encode(text) for text in texts]

    def batch_decode(self, tokens: List[List[int]], **kwargs) -> List[str]:
        """
        Decode a batch of token ID lists back into strings.

        Args:
            tokens (List[List[int]]): A list of token ID lists to decode.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            List[str]: The list of decoded strings.
        """
        return [self.encoding.decode(token_list) for token_list in tokens]

    @property
    def vocab_size(self) -> int:
        """
        Get the size of the tokenizer's vocabulary.

        Returns:
            int: Vocabulary size.
        """
        return self.encoding.n_vocab

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the tokenizer's vocabulary.

        Note: `tiktoken` does not expose a direct method to retrieve the vocabulary.
        This function raises a NotImplementedError.

        Raises:
            NotImplementedError: Always, since `tiktoken` does not support this.

        Returns:
            Dict[str, int]: Vocabulary dictionary.
        """
        raise NotImplementedError(
            "tiktoken does not support retrieving the full vocabulary."
        )

    def save_pretrained(self, save_directory: str):
        """
        Save the tokenizer configuration. This is a stub since `tiktoken` does not support saving.

        Args:
            save_directory (str): Directory to save the tokenizer configuration.
        """
        raise NotImplementedError(
            "tiktoken does not support saving tokenizer configurations."
        )

    @classmethod
    def from_pretrained(cls, model_name: str):
        """
        Load the tokenizer from a pretrained model.

        Args:
            model_name (str): The name of the OpenAI model.

        Returns:
            TiktokenHuggingFaceTokenizer: An instance of the tokenizer.
        """
        return cls(model_name)

    def apply_chat_template(self, template: str, **kwargs) -> str:
        """
        Apply a chat template to a prompt.

        Args:
            template (str): The chat template to apply.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            str: The prompt with the chat template applied.
        """
        return template
